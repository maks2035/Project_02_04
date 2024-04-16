#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/features2d.hpp>
#include <mpi.h>
#include <iostream>
#include <vector>

std::string get_name(const cv::Mat& image, const std::vector<cv::Mat>& etalons, const std::vector<cv::Mat>& etalons_descriptors, const std::vector<std::string>& names) {
   cv::Ptr<cv::FlannBasedMatcher> matcher = cv::FlannBasedMatcher::create();

   cv::Ptr<cv::Feature2D> sift = cv::SIFT::create();
   std::vector<cv::KeyPoint> kp;
   cv::Mat dis;
   sift->detectAndCompute(image, cv::noArray(), kp, dis);

   double min_dist = DBL_MAX;
   int min_dist_idx = -1;

   for (size_t i = 0; i < etalons.size(); ++i) {
      std::vector<cv::DMatch> matches;
      matcher->match(dis, etalons_descriptors[i], matches);

      double dist = 0;
      for (size_t j = 0; j < matches.size(); ++j) {
         dist += matches[j].distance;
      }
      dist /= matches.size();

      if (dist < min_dist) {
         min_dist = dist;
         min_dist_idx = i;
        }
   }

   if (min_dist_idx != -1) {
      return names[min_dist_idx];
   }
   else {
      return "Unknown";
   }
}

void rot(cv::RotatedRect& box, cv::Mat& image, cv::Mat& cropped) {
   cv::Mat rotated;

   cv::Size rect_size = box.size;
   if (box.angle < -45.) {
      std::swap(rect_size.width, rect_size.height);
      box.angle += 90.0;
   }

   cv::Mat M = cv::getRotationMatrix2D(box.center, box.angle, 1.0);
   cv::warpAffine(image, rotated, M, image.size(), cv::INTER_CUBIC);

   cv::getRectSubPix(rotated, rect_size, box.center, cropped);
   if (cropped.size().width > cropped.size().height) {
      cv::rotate(cropped, cropped, cv::ROTATE_90_CLOCKWISE);
   }
}

int main(int argc, char** argv){
   MPI_Init(&argc, &argv);

   int rank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   std::vector<std::string> names = { "10 of clubs", "8 of hearts", "Jack of Diamonds", "Queen of spades", "Ace of diamonds", "Jack of clubs"};
   std::vector<cv::Mat> etalons;
   std::vector<std::vector<cv::KeyPoint>> etalons_keypoints;
   std::vector<cv::Mat> etalons_descriptors;
   for (int i = 1; i <= 6; i++) {
      cv::Mat etalon = cv::imread("D:/virandfpc/vir/Project_02_04/" + std::to_string(i) + ".jpg");
      etalons.push_back(etalon);
      cv::Ptr<cv::Feature2D> sift = cv::SIFT::create();
      std::vector<cv::KeyPoint> kp;
      cv::Mat dis;
      sift->detectAndCompute(etalon, cv::noArray(), kp, dis);
      etalons_keypoints.push_back(kp);
      etalons_descriptors.push_back(dis);
   }

   cv::Mat image = cv::imread("D:/virandfpc/vir/Project_02_04/7.jpg");
   cv::resize(image, image, cv::Size(), 0.5, 0.5);
   cv::Mat image_clone = image.clone();

   cv::VideoCapture cap("D:/virandfpc/vir/Project_02_04/1.mp4");
   if (!cap.isOpened()) {
      std::cout << "Error: could not open video writer" << std::endl;
      MPI_Finalize();
      return -1;
   }

   std::vector<cv::Mat> frames;
   int frame_count = 0;
   while (true) {
      cv::Mat frame;
      cap >> frame;
      if (frame.empty()) break;

      if (frame_count % size == rank) {
         cv::resize(frame, frame, cv::Size(), 0.5, 0.5);
         cv::Mat frame_clone = frame.clone();

         cv::Mat  gauss_frame, edges_frame;
         GaussianBlur(frame, gauss_frame, cv::Size(11, 11), 0);
         Canny(gauss_frame, edges_frame, 100, 150);

         std::vector<cv::Vec4i> hierarchy;
         std::vector<std::vector< cv::Point>> contours;
         findContours(edges_frame, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

         for (size_t i = 0; i < contours.size(); i++) {

            double epsilon = 0.02 * arcLength(contours[i], true);
            std::vector<cv::Point> approx;
            approxPolyDP(contours[i], approx, epsilon, true);

            cv::RotatedRect box = cv::minAreaRect(approx);

            cv::Mat img;
            rot(box, frame, img);

            if (img.empty()) {
               continue;
            }

            std::string name = get_name(img, etalons, etalons_descriptors, names);

            if (approx.size() == 4 && cv::isContourConvex(approx)) {
               cv::Scalar color = cv::Scalar(0, 255, 0);
               drawContours(frame_clone, contours, (int)i, color, 1, cv::LINE_8, hierarchy, 0);

               cv::Moments M = moments(approx);
               cv::Point center(M.m10 / M.m00, M.m01 / M.m00);
               putText(frame_clone, name, center, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.65, color, 1);
            }
         }
         frames.push_back(frame_clone.clone());
      }
      frame_count++;
   }

   cap.release();

   if (rank == 0) {

      cv::VideoWriter video("D:/virandfpc/vir/Project_02_04/output.mp4", cv::VideoWriter::fourcc('a', 'v', 'c', '1'), 30, cv::Size(480, 360));
      if (!video.isOpened()) {
         std::cout << "Error: could not open video writer" << std::endl;
         return -1;
      }

      for (int i = 0; i < frames.size(); i++) {
         cv::imshow("video", frames[i]);
         cv::Mat buff = frames[i].clone();
         cv::resize(buff, buff, cv::Size(480, 360));
         video << buff;

         char c = (char)cv::waitKey(30);
         if (c == 27) break;
      }

      cv::destroyAllWindows();
      video.release();

   }

   MPI_Finalize();

   return 0;

}