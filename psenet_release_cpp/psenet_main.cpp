/****************************************************************************************
********
********   file name:   psenet_main.cpp
********   description: psenet torchlib test cpp code
********   version:     V1.0
********   author:      He Wen
********   time:        2020-05-22 09:54
********
*****************************************************************************************/


#include <iostream>
#include "torch/script.h"
#include "torch/torch.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <vector>
#include <stdio.h>
#include <map>
#include <algorithm>

using namespace std;

#define PC_DEBUG

void growing_text_line(vector<cv::Mat> &kernals, vector<vector<int>> &text_line, float min_area);

int main()
{
    torch::DeviceType device_type;
    if (torch::cuda::is_available())
    {
        device_type = torch::kCUDA;
    }
    else
    {
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    torch::jit::script::Module module = torch::jit::load("../model/psenet.pt");//加载pytorch模型
    module.to(device);
    std::cout<<"load model success"<<std::endl;

    cv::Mat image = cv::imread("../pic/11.jpg");//读取图片
    if(image.empty())
    {
        printf("please check out image path\n");
    }

    cv::Mat img_float, img_rgb;
    cv::cvtColor(image, img_rgb, CV_BGR2RGB);
    int imgw = image.cols;
    int imgh = image.rows;
    int input = 1120;
    float scale = input/(float)(imgh>imgw?imgh:imgw);  //2240为python测试代码中的输入尺寸,因psenet为全卷积网络,与全连接层不一样的是其输入尺寸可不固定大小
    std::cout<<"input w: "<<input<<std::endl;
    int outimgw = scale*imgw;
    int outimgh = scale*imgh;
    cv::resize(img_rgb, img_rgb, cv::Size(outimgw, outimgh));
    img_rgb.convertTo(img_float, CV_32F, 1.0f / 255.0f);   //归一化到[0,1]区间

    auto tensor_image = torch::from_blob(img_float.data, {1, outimgh, outimgw, 3}, torch::kFloat32);  //对于一张图而言可使用此函数将nhwc格式转换成tensor
    tensor_image = tensor_image.permute({0, 3, 1, 2});//调整通道顺序,将nhwc转换成nchw
    tensor_image = tensor_image.to(at::kCUDA);   //将tensor放进GPU中处理

    torch::Tensor out_tensor = module.forward({tensor_image}).toTensor();  //前向计算

    long time1 = static_cast<double>( cv::getTickCount());  //后处理开始时间
    out_tensor = out_tensor.squeeze().detach().permute({1, 2, 0});  //这条语句包含了三个处理处理函数,处理完成后将N*C*H*W的tensor转换成H*W*C格式tensor
                                                                    //1.squeeze()对tensor进行降维,丢弃值为1的维度   C*H*W
                                                                    //2.detach()将variable参数从网络中隔离开，不参与参数更新
                                                                    //3.permute()调整通道顺序  H*W*C
    out_tensor = out_tensor.mul(255).clamp(0, 255).to(torch::kU8);  //因预处理阶段将输入图像做了归一化处理,现在需要将tensor还原到0,255,
                                                                    //mul(255),tensor乘以255  clamp(0, 255)tensor内小于0的值置为0,大于255的值置为255
    out_tensor = out_tensor.to(torch::kCPU);   //将tensor放进CPU中处理
    cv::Mat resultImg(outimgh, outimgw, CV_8UC3);
    std::memcpy((void *) resultImg.data, out_tensor.data_ptr(), sizeof(torch::kU8) * out_tensor.numel());//将tensor的数据拷贝到mat类型的resultImg中  numel()获取像素个数

    cv::Mat  splitChannels[3];    //psenet前向计算后输出的结果为三张目标尺度不同的图像
    vector<cv::Mat> resultimg_vec;
    for(int i=0;i<3;i++)
    {
        cv::Mat thre;
        cv::split(resultImg,splitChannels);
        cv::threshold(splitChannels[i], thre, 126, 255, cv::THRESH_BINARY);
        resultimg_vec.push_back(thre);
    }

#ifdef PC_DEBUG
    cv::Mat channe0, channe1, channe2;
    splitChannels[0].copyTo(channe0);
    splitChannels[1].copyTo(channe1);
    splitChannels[2].copyTo(channe2);
    cv::imwrite("./channe0.jpg", channe0);
    cv::imwrite("./channe1.jpg", channe1);
    cv::imwrite("./channe2.jpg", channe2);
#endif

    cv::Mat temp;
    cv::Mat img(resultImg.rows, resultImg.cols, CV_8UC1);
    cv::Mat out(img.size(), img.type(), cv::Scalar(255));  //创建纯白图,将growing_text_line()的输出text_line保存为mat类型

    vector<vector<int>> text_line;
    float min_w_h = std::min(resultImg.cols, resultImg.rows);
    min_w_h *= min_w_h / 50;    //过滤掉小于图像面积50分之一的连通域

    growing_text_line(resultimg_vec, text_line, min_w_h);//psenet网络的算法关键,基于渐进式尺寸可扩展网络的形状鲁棒文本检测 ,所谓渐进式尺寸类似于图像膨胀处理
    for(int i=0;i<text_line.size();i++)
    {
        int j = 0;
        for(vector<int>::iterator iter = text_line[i].begin();iter != text_line[i].end();iter++)
        {
            if(*iter != 0)  //*iter表示像素的id(属于哪个label,0为背景,非0为连通域)
            {
                out.at<uchar>(i,j) = 0;  //将非背景的像素点灰度值置为0,目标区域为黑色,背景为白色
            }
            j++;
        }
    }

    cv::threshold(out, temp, 126, 255, cv::THRESH_BINARY_INV); //二值化图像,后面的处理就是找到各连通域的最小外接矩形
#ifdef PC_DEBUG
    cv::imwrite("./temp_thresh.jpg", temp);
#endif

    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> hierarcy;
    cv::findContours(temp, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    vector<cv::Rect> boundRect(contours.size());  //定义外接矩形集合
    vector<cv::RotatedRect> box(contours.size()); //定义最小外接矩形集合
    cv::Point2f rect[4];
    std::cout<<"contours num: "<<contours.size()<<std::endl;
    for (int i = 0; i < contours.size(); i++)
    {
        box[i] = cv::minAreaRect(cv::Mat(contours[i]));  //计算每个轮廓最小外接矩形
        boundRect[i] = cv::boundingRect(cv::Mat(contours[i]));
        box[i].points(rect);  //把最小外接矩形四个端点复制给rect数组
        for(int j=0; j<4; j++)
        {
            cv::line(image, cv::Point(rect[j].x/scale, rect[j].y/scale), cv::Point(rect[(j+1)%4].x/scale, rect[(j+1)%4].y/scale),cv::Scalar(0, 0, 255), 2, 8);  //绘制最小外接矩形每条边
        }
    }

#ifdef PC_DEBUG
    cv::imwrite("./image_result.jpg", image);
#endif
    
    std::cout << "后处理时间:" << (static_cast<double>( cv::getTickCount()) - time1) / cv::getTickFrequency() << "s" << std::endl;
    return 0;
}

void growing_text_line(vector<cv::Mat> &kernals, vector<vector<int>> &text_line, float min_area)
{
    cv::Mat label_mat;
    // 第一步：寻找连通域
    int label_num = cv::connectedComponents(kernals[kernals.size() - 1], label_mat, 4);
    int area[label_num + 1];//统计每个文字块像素的个数即面积
    memset(area, 0, sizeof(area));
    for (int x = 0; x < label_mat.rows; ++x)
    {
        for (int y = 0; y < label_mat.cols; ++y)
        {
            int label = label_mat.at<int>(x, y);
            if (label == 0) continue;
            area[label] += 1;
        }
    }

    std::queue<cv::Point> queue, next_queue;//重要：队列，先进先出
    for (int x = 0; x < label_mat.rows; ++x)
    {
        vector<int> row(label_mat.cols);
        for (int y = 0; y < label_mat.cols; ++y)
        {
            int label = label_mat.at<int>(x, y);

            if (label == 0) continue;
            if (area[label] < min_area) continue;

            cv::Point point(x, y);
            queue.push(point);//重要：队列保存非0位置
            row[y] = label;//非0的label保存
        }
        text_line.emplace_back(row);
    }
    // text_line： 传出去的text_line先保存了最瘦的那个分割图各个label

    // 第一步：寻找连通域 END
    // 第二、三步：像素点扩展
    //4邻域
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};
    // 从倒数第二个开始，因为是以倒数第一个最瘦的为基础的

    for (int kernal_id = kernals.size() - 2; kernal_id >= 0; --kernal_id)
    {
        while (!queue.empty())
        {
            // 出队
            cv::Point point = queue.front(); queue.pop();
            int x = point.x;
            int y = point.y;
            int label = text_line[x][y];

            bool is_edge = true;
            for (int d = 0; d < 4; ++d)
            {
                int tmp_x = x + dx[d]; //对x位置像素进行逐点膨胀
                int tmp_y = y + dy[d];

                if (tmp_x < 0 || tmp_x >= (int)text_line.size()) continue; //当tmp_x<0,或tmp_x超过或等于图像的宽度则进行下一次循环
                if (tmp_y < 0 || tmp_y >= (int)text_line[1].size()) continue;//当tmp_y<0,或tmp_y超过或等于图像的高度则进行下一次循环
                if (kernals[kernal_id].at<char>(tmp_x, tmp_y) == 0) continue;//当连通域为0,即连通域为空的时候,进行下一次循环
                if (text_line[tmp_x][tmp_y] > 0) continue;  //若当前膨胀后的kernel与其他kernel接触时,则认定此kernel到达了目标的边界,此时进行下一次循环
                /*
                // 能够下来的需要满足两个条件：
                //1. (kernals[kernal_id].at<char>(tmp_x, tmp_y) != 0)
                //2. (text_line[tmp_x][tmp_y] == 0)
                // 1. 上个分割图对应位置上有label
                //2. 本位置无label
                //对应上图中的灰色部分 属于S2中的kernel的但不属于S1中的kernel的像素点
                // 满足这两个条件就放到队列最后（queue.push(point)）;，同时把该位置归化为自己的label（ text_line[tmp_x][tmp_y] = label;）
                */
                cv::Point point(tmp_x, tmp_y);
                queue.push(point);
                text_line[tmp_x][tmp_y] = label;
                is_edge = false;
            }
            if (is_edge)
            {
                next_queue.push(point);
            }
        }
        swap(queue, next_queue);
    }
}

cv::Mat draw_bbox(cv::Mat &src, const std::vector<std::vector<cv::Point>> &bboxs)
{
    cv::Mat dst;
    if (src.channels() == 1)
    {
        cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);
    } else
    {
        dst = src.clone();
    }
    auto color = cv::Scalar(0, 0, 255);
    for (auto bbox :bboxs)
    {
        cv::line(dst, bbox[0], bbox[1], color, 3);
        cv::line(dst, bbox[1], bbox[2], color, 3);
        cv::line(dst, bbox[2], bbox[3], color, 3);
        cv::line(dst, bbox[3], bbox[0], color, 3);
    }
    return dst;
}

//reference: https://blog.csdn.net/xiamentingtao/article/details/98673967

