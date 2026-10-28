#include <cmath>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include "core/face_mask_generator.h"
#include "model/output_processor.h"

int main() {
    int failures = 0;
    auto check = [&](bool ok, const char* description) {
        std::cout << (ok ? "PASS " : "FAIL ") << description << '\n';
        if (!ok) ++failures;
    };
    std::vector<cv::Point2f> points(68, {48, 48});
    for (int i=0; i<20; ++i) {
        const double angle = i * 2 * 3.141592653589793 / 20;
        points[48+i] = {float(48+14*std::cos(angle)), float(68+6*std::sin(angle))};
    }
    digital_human::core::FaceMaskGenerator masks;
    auto legacy = masks.generatePreciseMouthAlphaMask96(points);
    auto explicit_legacy = masks.generatePreciseMouthAlphaMask96(points, 0);
    auto outward = masks.generatePreciseMouthAlphaMask96(points, 3);
    check(cv::norm(legacy, explicit_legacy, cv::NORM_INF) == 0, "default mask keeps historical behavior");
    double minv=0, maxv=0;
    cv::minMaxLoc(outward, &minv, &maxv);
    check(outward.type()==CV_32FC1 && minv>=0 && maxv<=1, "outward alpha is bounded float32");
    check(outward.at<float>(68,48)==1, "mouth center remains fully opaque");
    check(outward.at<float>(0,0)==0, "distant background remains transparent");
    check(cv::countNonZero((outward>0) & (outward<1))>0, "outward feather has a smooth transition band");

    cv::Mat original(192,192,CV_8UC3,cv::Scalar(255,0,0));
    ncnn::Mat generated(96,96,3);
    generated.channel(0).fill(1.f);
    generated.channel(1).fill(0.f);
    generated.channel(2).fill(0.f);
    cv::Mat mask(96,96,CV_32FC1,cv::Scalar(1));
    cv::Mat inverse=(cv::Mat_<double>(2,3)<<1,0,48,0,1,48);
    digital_human::model::OutputProcessor output;
    auto a=output.ProcessROI(generated,original,mask,inverse,{48,48,96,96});
    output.SetGeneratedBlendWeight(1.f);
    auto b=output.ProcessROI(generated,original,mask,inverse,{48,48,96,96});
    auto full=output.Process(generated,original,mask,inverse);
    check(!a.empty() && !b.empty() && !full.empty(), "both rendering paths return images");
    const auto old=a.at<cv::Vec3b>(96,96), direct=b.at<cv::Vec3b>(96,96);
    check(old[0]>=70 && old[0]<=80 && old[2]>=170 && old[2]<=181,
          "baseline retains approximately 30 percent original blue mouth");
    check(direct[0]<=1 && direct[2]>=250, "direct mode does not reintroduce original mouth pixels");
    check(full.at<cv::Vec3b>(96,96)[0]<=1 && full.at<cv::Vec3b>(96,96)[2]>=250,
          "full-frame fallback honors direct mode too");
    check(b.at<cv::Vec3b>(0,0)==original.at<cv::Vec3b>(0,0), "pixels outside the patch remain unchanged");
    return failures ? 1 : 0;
}
