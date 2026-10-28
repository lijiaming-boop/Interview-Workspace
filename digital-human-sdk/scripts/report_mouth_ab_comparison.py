"""Build evidence and a side-by-side viewer for the GPU mouth-fusion experiment."""
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import zipfile


def main():
    root = Path(sys.argv[1]).resolve()
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env['PATH'] = 'D:/MSYC64/ucrt64/bin;' + env.get('PATH', '')
    state = json.loads((root/'suite_status.json').read_text())
    assert state['state'] == 'completed' and len(state['results']) == 12
    variants = ['baseline', 'direct', 'outer-feather']
    labels = ['A baseline', 'B direct', 'C outer feather']
    descriptions = ['原有70%混合 + 11×11羽化', '取消旧嘴额外叠加 + 原羽化', '取消旧嘴叠加 + 核心不透明/向外羽化3px']
    summary, validation = [], []
    for v in variants:
        runs = [r for r in state['results'] if r['render_variant']==v and not r['encode']]
        video = next(r for r in state['results'] if r['render_variant']==v and r['encode'])
        summary.append({'variant':v, 'core_fps_median':statistics.median(r['pipeline_fps'] for r in runs),
            'core_fps_min':min(r['pipeline_fps'] for r in runs), 'core_fps_max':max(r['pipeline_fps'] for r in runs),
            'render_mean_ms_median':statistics.median(r['stages']['render_fusion']['mean_ms'] for r in runs),
            'inference_mean_ms_median':statistics.median(r['stages']['model_inference']['mean_ms'] for r in runs),
            'encoded_pipeline_fps':video['pipeline_fps'], 'encoded_render_mean_ms':video['stages']['render_fusion']['mean_ms'],
            'output_frames':video['output_frames'], 'expected_frames':video['expected_frames']})
        p = root/(v+'_video')
        probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(p/'lipsync.mp4')],env=env))
        vs = next(s for s in probe['streams'] if s['codec_type']=='video')
        au = next(s for s in probe['streams'] if s['codec_type']=='audio')
        assert vs['codec_name']=='h264' and au['codec_name']=='aac'
        assert int(vs['nb_frames'])==video['output_frames']==746
        assert vs['avg_frame_rate']=='25/1' and video['av_match_misses']==4
        timeline = list(csv.DictReader((p/'output_timeline.csv').open()))
        assert all(int(row['pts_ms'])==i*40 for i,row in enumerate(timeline))
        assert video['gpu_enabled'] and video['stages']['model_inference']['count']==746
        (p/'ffprobe.json').write_text(json.dumps(probe,indent=2),encoding='utf-8')
        validation.append({'variant':v,'gpu_enabled':True,'video_frames':746,'target_frames':750,
                           'duration_s':float(vs['duration']),'pts_contiguous':True,'tail_match_misses':4,
                           'lossless_snapshots':len(list((p/'snapshots').glob('*.png')))})
    for r in state['results']:
        assert r['gpu_enabled'] and r['output_frames']==746 and r['av_match_misses']==4
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (root/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    (root/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    filters=[]
    for i,label in enumerate(labels):
        filters.append(f"[{i}:v]crop=224:168:490:640,pad=iw:ih+28:0:28:color=0x111827,drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='{label}':fontcolor=white:fontsize=16:x=8:y=5[p{i}]")
    filters.append('[p0][p1][p2]hstack=inputs=3[out]')
    graph=';'.join(filters)
    for pts in [6000,12000,24000]:
        cmd=['ffmpeg','-nostdin','-v','error','-y']
        for v in variants:cmd+=['-threads','1','-i',str(root/(v+'_video')/f'snapshots/frame_{pts}.png')]
        cmd+=['-filter_complex_threads','1','-filter_complex',graph,'-map','[out]','-frames:v','1','-threads','1',str(root/f'comparison_{pts}.png')]
        subprocess.run(cmd,env=env,check=True)
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(root/'inputs/face3.png'),
        '-vf','crop=224:168:490:640','-frames:v','1','-threads','1',str(root/'original_mouth.png')],env=env,check=True)
    cmd=['ffmpeg','-nostdin','-v','error','-y']
    for v in variants:cmd+=['-threads','1','-i',str(root/(v+'_video')/'lipsync.mp4')]
    cmd+=['-filter_complex_threads','1','-filter_complex',graph,'-map','[out]','-map','0:a:0',
          '-c:v','libx264','-preset','veryfast','-crf','16','-pix_fmt','yuv420p','-threads','2',
          '-c:a','copy','-movflags','+faststart',str(root/'mouth_comparison.mp4')]
    subprocess.run(cmd,env=env,check=True,timeout=120)
    comparison_probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format',
        '-of','json',str(root/'mouth_comparison.mp4')],env=env))
    comparison_video = next(s for s in comparison_probe['streams'] if s['codec_type']=='video')
    assert int(comparison_video['nb_frames'])==746
    (root/'comparison_ffprobe.json').write_text(json.dumps(comparison_probe,indent=2),encoding='utf-8')
    lines=['# GPU嘴部回贴三组对照','','2026-10-27。测试3，1024×1536；原曲60–90秒片段；RTX 4050 Vulkan；同一Wav2Lip权重、对齐、锐化、插值与H.264/AAC编码参数。',
        '', '三次无编码计时采用ABC/BCA/CAB顺序；表中纯流水线FPS和渲染耗时取三次中位数。另各生成一段视频；无损PNG在计时结束后写出。', '',
        '|组别|配置|纯流水线FPS|渲染均值ms|带编码FPS|','|---|---|---:|---:|---:|']
    for label,desc,r in zip(labels,descriptions,summary):
        lines.append(f"|{label}|{desc}|{r['core_fps_median']:.2f}|{r['render_mean_ms_median']:.3f}|{r['encoded_pipeline_fps']:.2f}|")
    lines += ['', '## 画面对比与结论', '',
        '固定查看片段第6/12/24秒的原尺寸无损裁剪（对应原曲66/72/84秒），没有对嘴部额外放大、锐化或AI修复。左到右为A/B/C。', '',
        '![第12秒嘴部对照](comparison_12000.png)', '',
        '- A保留了原图的部分高频纹理，当前这张底图的融合观感更协调，但仍有低分辨率模糊。',
        '- B取消旧嘴叠加后，没有出现新的可辨认唇纹或牙齿细节；低分辨率生成区域及色差更显眼。',
        '- C让嘴部核心完全采用生成图，外圈羽化更窄，但也暴露了更明显的模糊贴片和唇周过渡，当前不建议作为默认。',
        '- 因此，本轮证据不支持“取消叠加或缩小羽化就能显著提高嘴部清晰度”。这些处理主要改变融合权重，96×96整脸/约28×13嘴部的原生信息量没有增加。',
        '- B/C的渲染略快，是因为跳过了最终的逐像素颜色混合，不能把速度提高当成画质提高。',
        '- 结论只适用于这张底图和该音频片段的抽检；没有客观高清真值，未以锐度分数冒充真实细节恢复，也未完成逐帧人工审片。',
        '', '默认配置仍为A；B/C是显式可选实验参数。当前适合保留A，再对人脸修复或高分辨率口型模型做下一轮验证。', '',
        '## 已知问题', '',
        '三组均输出746/750帧（29.84秒），当前AV匹配器末尾4帧失败，未修改或补帧。三组PTS一致，口型对比时间点一致。',
        '相关回归测试4项均通过，见[regression.log](regression.log)。', '',
        '## 产物', '', '[交互对照页](comparison.html) · [嘴部三组同屏视频](mouth_comparison.mp4) · [CSV](summary.csv) · [验证记录](validation.json)', '',
        '[A完整画面](baseline_video/lipsync.mp4) · [B完整画面](direct_video/lipsync.mp4) · [C完整画面](outer-feather_video/lipsync.mp4)', '',
        '[6秒无损对照](comparison_6000.png) · [12秒无损对照](comparison_12000.png) · [24秒无损对照](comparison_24000.png) · [原图嘴部](original_mouth.png)', '',
        '[输入与源码清单](manifest.json) · [原始运行结果](suite_status.json) · [源码快照](source_snapshot.zip)', '']
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    cards=[]
    for label,desc,r in zip(labels,descriptions,summary):
        v=r['variant']; cards.append(f'<article><h2>{label}</h2><p>{desc}</p><p>核心 {r["core_fps_median"]:.2f} FPS · 渲染 {r["render_mean_ms_median"]:.3f} ms</p><video controls preload="none" poster="{v}_video/snapshots/frame_12000.png" src="{v}_video/lipsync.mp4"></video></article>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPU嘴部回贴 A/B/C</title>
<style>body{font:16px/1.6 system-ui;max-width:1200px;margin:32px auto;padding:0 20px;background:#111827;color:#edf2f7}a{color:#7dd3fc}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}article{background:#1f2937;border-radius:10px;padding:14px}video{width:100%;background:#080d16}article video{height:450px;object-fit:contain}img{max-width:100%}.note{padding:14px;background:#422e15;border-radius:8px}button{padding:8px 16px;cursor:pointer}@media(max-width:700px){.cards{grid-template-columns:1fr}}</style>
<h1>GPU嘴部回贴三组对照</h1><p>测试3 · 原曲60–90秒 · 模型/对齐/锐化/编码相同</p>
<p class="note">当前这组素材中，B/C没有恢复真实细节，贴片和色差反而更显眼；默认仍为A。三组都保留了原有末尾4帧匹配失败。</p>
<p><a href="REPORT.md">报告</a> · <a href="summary.csv">CSV</a> · <a href="original_mouth.png">原图嘴部</a></p>
<h2>嘴部同屏视频（从左到右A/B/C）</h2><video controls preload="metadata" poster="comparison_12000.png" src="mouth_comparison.mp4"></video>
<h2>原尺寸无损对照</h2><p>片段第6秒</p><img src="comparison_6000.png"><p>片段第12秒</p><img src="comparison_12000.png"><p>片段第24秒</p><img src="comparison_24000.png">
<h2>完整画面</h2><button onclick="playAll()">从头同步播放三组</button><div class="cards">'''
    page+=''.join(cards)+'''</div><script>function playAll(){document.querySelectorAll('.cards video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{});});}</script></html>'''
    (root/'comparison.html').write_text(page,encoding='utf-8')
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    manifest['model_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [repo/'models/Wav2Lip-SD-GAN-opt.param',repo/'models/Wav2Lip-SD-GAN-opt.bin']}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    with zipfile.ZipFile(root/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for directory in ('src','include','examples','scripts','cmake'):
            for p in (repo/directory).rglob('*'):
                if p.is_file() and p.suffix in ('.cpp','.h','.py','.cmake','.in','.txt'): archive.write(p,p.relative_to(repo))
        for n in ('CMakeLists.txt','CMakePresets.json','CMakeUserPresets.json'):archive.write(repo/n,n)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
