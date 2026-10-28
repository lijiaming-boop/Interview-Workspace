"""数据填充执行器

纯机械执行模块,不做智能决策,完全信任 LLM Skill 的填充建议

P1 改造（改进方案第五节 4/5）：
- 完整率分母按提取阶段标记的 available_fields 计算（模板本身没有的字段
  不再被算作"数据缺失"），与提取/质检阶段口径一致；
- 填充元数据携带默认值版本（defaults_version）与逐月填充明细，
  使"行业默认值口径变化"可追溯到最终节能率结果。
"""
import json
from pathlib import Path
from datetime import datetime

# 全部 7 个月度字段（available_fields 缺失时的兜底，与旧行为一致）
ALL_MONTHLY_FIELDS = [
    "total_energy", "chiller_supply_temp",
    "cooling_pump_freq", "cooling_water_temp_diff",
    "freezing_pump_freq", "freezing_water_temp_diff",
    "cooling_tower_outlet_temp",
]


class DataFiller:
    """数据填充器 - 执行 LLM 决策的填充操作"""

    def fill_data(
        self,
        extracted_data_path: str,
        filling_suggestions_path: str,
        output_path: str
    ) -> str:
        """执行数据填充

        Args:
            extracted_data_path: 提取的原始数据 JSON 路径
            filling_suggestions_path: LLM 生成的填充建议 JSON 路径
            output_path: 填充后数据输出路径

        Returns:
            输出文件路径
        """
        # 读取数据
        with open(extracted_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        with open(filling_suggestions_path, 'r', encoding='utf-8') as f:
            suggestions = json.load(f)

        # 为每个月度数据初始化 _filled_fields
        for month_data in data["monthly_data"]:
            month_data["_filled_fields"] = []

        # 完整率分母：优先用提取阶段标记的 available_fields（模板可用字段），
        # 避免"模板本身没有的字段"被算进缺失分母（P1 口径统一修复）
        available_fields = data.get("available_fields") or ALL_MONTHLY_FIELDS
        monthly_fields = [f for f in ALL_MONTHLY_FIELDS if f in available_fields]
        total_fields = len(data["monthly_data"]) * len(monthly_fields)
        monthly_filled_count = 0  # 仅统计月度字段填充（完整率口径）
        total_filled_count = 0    # 含顶层字段（如 electricity_price）的全部填充数

        # 定义受保护字段（不允许填充建议修改）
        protected_fields = [
            "project_region",           # 项目地区（由提取阶段确定）
            "station_type",             # 站房类型（由提取阶段检测）
            "station_type_confidence",  # 站房类型置信度
            "station_type_reason"       # 站房类型识别依据
        ]

        # 执行填充
        for suggestion in suggestions.get("fillable_suggestions", []):
            field_path = suggestion["field"]
            suggested_value = suggestion["suggested_value"]

            # 跳过受保护字段（防止错误修改关键信息）
            if field_path in protected_fields:
                print(f"Warning: Skipping protected field: {field_path}")
                print(f"  Suggested value: {suggested_value}")
                print("  Reason: This field should only be set by extract-and-check-data skill")
                continue

            # 解析字段路径: "monthly_data[1].chiller_supply_temp"
            if field_path.startswith("monthly_data["):
                month_idx = int(field_path.split('[')[1].split(']')[0])
                field_name = field_path.split('.')[1]

                month_data = data["monthly_data"][month_idx]

                # 只填充 null 值,不覆盖已有数据
                if month_data.get(field_name) is None:
                    month_data[field_name] = suggested_value
                    month_data["_filled_fields"].append(field_name)
                    monthly_filled_count += 1
                    total_filled_count += 1
            else:
                # 顶层字段（如 electricity_price）
                if data.get(field_path) is None:
                    data[field_path] = suggested_value
                    total_filled_count += 1

        # 计算最终完整率（分母与上面一致：按可用月度字段计）
        # P1 修复：完整率只与月度字段相关——顶层字段（electricity_price 等）
        # 的填充数不再计入月度分母，否则原始完整率会偏低甚至为负。
        null_count = sum(
            1 for month in data["monthly_data"]
            for field in monthly_fields
            if month.get(field) is None
        )

        if total_fields > 0:
            original_completeness = 1 - (null_count + monthly_filled_count) / total_fields
            final_completeness = 1 - null_count / total_fields
        else:
            original_completeness = final_completeness = 0.0

        # 添加元数据（P1：默认值版本 + 逐月填充明细，可追溯口径）
        data["_metadata"] = {
            "fill_timestamp": datetime.now().isoformat(),
            "fill_count": total_filled_count,
            "monthly_fill_count": monthly_filled_count,
            "original_completeness": round(original_completeness, 2),
            "final_completeness": round(final_completeness, 2),
            "completeness_denominator": {
                "available_fields": monthly_fields,
                "months": len(data["monthly_data"]),
                "total_fields": total_fields,
            },
            "defaults_version": suggestions.get("defaults_version", "unversioned"),
            "fill_detail": {
                str(idx): m["_filled_fields"]
                for idx, m in enumerate(data["monthly_data"]) if m["_filled_fields"]
            },
        }

        # 输出结果
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return str(output_path)
