#include <math.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "UDPSender.h"
#include "FaceEngine.h"
#include "nvAR.h"
#include "nvAR_defs.h"
#include "opencv2/opencv.hpp"
#include "RenderingUtils.h"


#ifndef M_PI
#define M_PI 3.1415926535897932385
#endif /* M_PI */
#ifndef M_2PI
#define M_2PI 6.2831853071795864769
#endif /* M_2PI */
#ifndef M_PI_2
#define M_PI_2 1.5707963267948966192
#endif /* M_PI_2 */
#define F_PI ((float)M_PI)
#define F_PI_2 ((float)M_PI_2)
#define F_2PI ((float)M_2PI)

#ifdef _MSC_VER
#define strcasecmp _stricmp
#endif /* _MSC_VER */

#define BAIL(err, code) \
  do {                  \
    err = code;         \
    goto bail;          \
  } while (0)

///
/// Definitions
/// 
/// 
/// 
bool FLAG_debug = false, FLAG_verbose = false, FLAG_temporal = true, FLAG_captureOutputs = false,
     FLAG_offlineMode = false, FLAG_isNumLandmarks126 = true;
std::string FLAG_outDir, FLAG_inFile, FLAG_outFile, FLAG_modelPath, FLAG_landmarks, FLAG_proxyWireframe,
    FLAG_captureCodec = "avc1", FLAG_camRes, FLAG_faceModel;
unsigned int FLAG_batch = 1, FLAG_appMode = 2;


static int StringToFourcc(const std::string &str) {
  union chint {
    int i;
    char c[4];
  };
  chint x = {0};
  for (int n = (str.size() < 4) ? (int)str.size() : 4; n--;) x.c[n] = str[n];
  return x.i;
}

enum {
  myErrNone = 0,
  myErrShader = -1,
  myErrProgram = -2,
  myErrTexture = -3,
};


#if 1
class MyTimer {
 public:
  void start() { t0 = std::chrono::high_resolution_clock::now(); }       /**< Start  the timer. */
  void pause() { dt = std::chrono::high_resolution_clock::now() - t0; }  /**< Pause  the timer. */
  void resume() { t0 = std::chrono::high_resolution_clock::now() - dt; } /**< Resume the timer. */
  void stop() { pause(); }                                               /**< Stop   the timer. */
  double elapsedTimeFloat() const {
    return std::chrono::duration<double>(dt).count();
  } /**< Report the elapsed time as a float. */
 private:
  std::chrono::high_resolution_clock::time_point t0;
  std::chrono::high_resolution_clock::duration dt;
};
#endif

class DoApp {
 public:
  enum Err {
    errNone           = FaceEngine::Err::errNone,
    errGeneral        = FaceEngine::Err::errGeneral,
    errRun            = FaceEngine::Err::errRun,
    errInitialization = FaceEngine::Err::errInitialization,
    errRead           = FaceEngine::Err::errRead,
    errEffect         = FaceEngine::Err::errEffect,
    errParameter      = FaceEngine::Err::errParameter,
    errUnimplemented,
    errMissing,
    errVideo,
    errImageSize,
    errNotFound,
    errFaceModelInit,
    errGLFWInit,
    errGLInit,
    errRendererInit,
    errGLResource,
    errGLGeneric,
    errFaceFit,
    errNoFace,
    errSDK,
    errCuda,
    errCancel,
    errCamera
  };
  Err doAppErr(FaceEngine::Err status) { return (Err)status; }
  FaceEngine face_ar_engine;
  DoApp();
  ~DoApp();

  void stop();
  Err initFaceEngine(const char *modelPath = nullptr, bool isLandmarks126 = false);
  Err initCamera(const char *camRes = nullptr, const int fps = 30);
  Err acquireFrame();
  Err acquireFaceBox();
  Err acquireFaceBoxAndLandmarks();
  Err fitFaceModel();
  Err run();
  void drawFPS(cv::Mat &img);
  void DrawBBoxes(const cv::Mat &src, NvAR_Rect *output_bbox);
  void DrawLandmarkPoints(const cv::Mat &src, NvAR_Point2f *facial_landmarks, int numLandmarks);
  void DrawFaceMesh(const cv::Mat &src, NvAR_FaceMesh *face_mesh);
  Err setProxyWireframe(const char *path);
  void processKey(int key);
  void getFPS();
  void init_sender(std::string& ip, int port);
  void send_data(double* buffer_data);
  static const char *errorStringFromCode(Err code);

  std::unique_ptr<UDPSender> sender;

  cv::VideoCapture cap{};
  cv::Mat frame;
  int inputWidth, inputHeight;
  int frameIndex;
  static const char windowTitle[];
  double frameTime;
  MyTimer frameTimer;
  cv::VideoWriter capturedVideo;

  FaceEngine::Err nvErr;
  float expr[6];
  bool drawVisualization, showFPS, captureVideo, captureFrame;
  std::vector<std::array<uint16_t, 3>> proxyWireframe;
  float scaleOffsetXY[4];
  std::vector<NvAR_Vector3u16> wfMesh_tvi_data;
};


///
/// 
/// 
/// 
/// 
/// 






void DoApp::init_sender(std::string& ip, int port)
{
    std::string temp_ip = ip;
    int temp_port = port;

    // Updata only if needed.
    if (this->sender)
    {
        if (temp_ip != this->sender->ip && temp_port != this->sender->port)
            return;
    }

    std::string ip_str = ip;
    int port_dest = port;

    ip_str = "127.0.0.1";

    if (port_dest == 0)
        port_dest = 4242;

    this->sender = std::make_unique<UDPSender>(ip_str.data(), port_dest);

}

void DoApp::send_data(double* buffer_data)
{
    sender->send_data(buffer_data);
}

DoApp *gApp = nullptr;
const char DoApp::windowTitle[] = "FaceTrack App";

void DoApp::processKey(int key) {
  switch (key) {
    case '3':
      face_ar_engine.destroyFeatures();
      face_ar_engine.setAppMode(FaceEngine::mode::faceMeshGeneration);
      nvErr =  face_ar_engine.createFeatures(FLAG_modelPath.c_str());
      // If there is an error, fallback to mode '2' i.e. landmark detection
      if (nvErr == FaceEngine::Err::errNone) {
        face_ar_engine.initFeatureIOParams();
        break;
      } else if (nvErr == FaceEngine::Err::errInitialization) {
      }
    case '2':
      face_ar_engine.destroyFeatures();
      face_ar_engine.setAppMode(FaceEngine::mode::landmarkDetection);
      face_ar_engine.createFeatures(FLAG_modelPath.c_str());
      face_ar_engine.initFeatureIOParams();
      break;
    case '1':
      face_ar_engine.destroyFeatures();
      face_ar_engine.setAppMode(FaceEngine::mode::faceDetection);
      face_ar_engine.createFeatures(FLAG_modelPath.c_str());
      face_ar_engine.initFeatureIOParams();
      break;
    case 'W':
    case 'w':
      drawVisualization = !drawVisualization;
      break;
    case 'F':
    case 'f':
      showFPS = !showFPS;
      break;
    default:
      break;
  }
}

DoApp::Err DoApp::initFaceEngine(const char *modelPath, bool isNumLandmarks126) {
  Err err = errNone;

  if (!cap.isOpened()) return errVideo;

  int numLandmarkPoints = isNumLandmarks126 ? 126 : 68;
  face_ar_engine.setNumLandmarks(numLandmarkPoints);

  nvErr = face_ar_engine.createFeatures(modelPath);
  if (nvErr != FaceEngine::Err::errNone) {
    if (nvErr == FaceEngine::Err::errInitialization && face_ar_engine.appMode == FaceEngine::mode::faceMeshGeneration) {
      //showFaceFitErrorMessage();
      printf("WARNING: face fitting has failed, trying to initialize Landmark Detection\n");
      face_ar_engine.destroyFeatures();
      face_ar_engine.setAppMode(FaceEngine::mode::landmarkDetection);
      nvErr = face_ar_engine.createFeatures(modelPath);
    }
  }

#ifdef DEBUG
  detector->setOutputLocation(outputDir);
#endif  // DEBUG

#define VISUALIZE
#ifdef VISUALIZE
  if (!FLAG_offlineMode) cv::namedWindow(windowTitle, 1);
#endif  // VISUALIZE

  frameIndex = 0;

  return doAppErr(nvErr);
}

void DoApp::stop() {
  face_ar_engine.destroyFeatures();


  cap.release();
#ifdef VISUALIZE
  cv::destroyAllWindows();
#endif  // VISUALIZE
}

void DoApp::DrawBBoxes(const cv::Mat &src, NvAR_Rect *output_bbox) {
  cv::Mat frm;
  if (FLAG_offlineMode)
    frm = src.clone();
  else
    frm = src;

  if (output_bbox)
    cv::rectangle(frm, cv::Point(lround(output_bbox->x), lround(output_bbox->y)),
                  cv::Point(lround(output_bbox->x + output_bbox->width), lround(output_bbox->y + output_bbox->height)),
                  cv::Scalar(255, 0, 0), 2);
}

void DoApp::DrawLandmarkPoints(const cv::Mat &src, NvAR_Point2f *facial_landmarks, int numLandmarks) {
  cv::Mat frm;
  if (FLAG_offlineMode)
    frm = src.clone();
  else
    frm = src;
  NvAR_Point2f *pt, *endPt;
  for (endPt = (pt = (NvAR_Point2f *)facial_landmarks) + numLandmarks; pt < endPt; ++pt)
    cv::circle(frm, cv::Point(lround(pt->x), lround(pt->y)), 1, cv::Scalar(0, 0, 255), -1);
  NvAR_Quaternion *pose = face_ar_engine.getPose();
  if (pose)
    face_ar_engine.DrawPose(frm, pose);
}

void DoApp::DrawFaceMesh(const cv::Mat &src, NvAR_FaceMesh *face_mesh) {
  cv::Mat render_frame = src;

  const cv::Scalar wfColor(0, 160, 0, 255);
  //Low resolution mesh can be used for visualization only
  if (proxyWireframe.size() > 0) {
    NvAR_FaceMesh wfMesh{nullptr, 0, nullptr, 0};
    wfMesh.num_vertices = face_mesh->num_vertices;
    wfMesh.vertices = face_mesh->vertices;
    wfMesh.num_tri_idx = proxyWireframe.size();
    wfMesh_tvi_data.resize(wfMesh.num_tri_idx);
    wfMesh.tvi = wfMesh_tvi_data.data();

    for (int i = 0; i < proxyWireframe.size(); i++) {
      auto proxyWireframe_idx = proxyWireframe[i];
      wfMesh.tvi[i].vec[0] = proxyWireframe_idx[0];
      wfMesh.tvi[i].vec[1] = proxyWireframe_idx[1];
      wfMesh.tvi[i].vec[2] = proxyWireframe_idx[2];
    }
    draw_wireframe(render_frame, wfMesh, *face_ar_engine.getRenderingParams(), wfColor);
  } else {
    draw_wireframe(render_frame, *face_mesh, * face_ar_engine.getRenderingParams(), wfColor);
  }

}

DoApp::Err DoApp::initCamera(const char* camRes, const int fps) {
    
    if (cap.open(0)) {
        if (camRes) {
            int n;
            n = sscanf(camRes, "%d%*[xX]%d", &inputWidth, &inputHeight);
            switch (n) {
            case 2:
                break;  // We have read both width and height
            case 1:
                inputHeight = inputWidth;
                inputWidth = (int)(inputHeight * (4. / 3.) + .5);
                break;
            default:
                inputHeight = 0;
                inputWidth = 0;
                break;
            }
            if (inputWidth) cap.set(CV_CAP_PROP_FRAME_WIDTH, inputWidth);
            if (inputHeight) cap.set(CV_CAP_PROP_FRAME_HEIGHT, inputHeight);

            inputWidth = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
            inputHeight = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);
            face_ar_engine.setInputImageWidth(inputWidth);
            face_ar_engine.setInputImageHeight(inputHeight);
        }
    }
    else
        return errCamera;
    return errNone;
}

DoApp::Err DoApp::acquireFrame() {
  Err err = errNone;

  // If the machine goes to sleep with the app running and then wakes up, the camera object is not destroyed but the
  // frames we try to read are empty. So we try to re-initialize the camera with the same resolution settings. If the
  // resolution has changed, you will need to destroy and create the features again with the new camera resolution (not
  // done here) as well as reallocate memory accordingly with FaceEngine::initFeatureIOParams()
  cap >> frame;  // get a new frame from camera into the class variable frame.
  if (frame.empty()) {
    // if in Offline mode, this means end of video,so we return
    if (FLAG_offlineMode) return errVideo;
    // try Init one more time if reading frames from camera
    err = initCamera(FLAG_camRes.c_str());
    if (err != errNone)
      return err;
    cap >> frame;
    if (frame.empty()) return errVideo;
  }

  return err;
}

DoApp::Err DoApp::acquireFaceBox() {
  Err err = errNone;
  NvAR_Rect output_bbox;

  // get landmarks in  original image resolution coordinate space
  unsigned n = face_ar_engine.acquireFaceBox(frame, output_bbox, 0);

  if (n && FLAG_verbose) {
    printf("FaceBox: [\n");
    printf("%7.1f%7.1f%7.1f%7.1f\n", output_bbox.x, output_bbox.y, output_bbox.x + output_bbox.width,
           output_bbox.y + output_bbox.height);
    printf("]\n");
  }

  if (0 == n) return errNoFace;

#ifdef VISUALIZE

  if (drawVisualization) {
    DrawBBoxes(frame, &output_bbox);
  }
#endif  // VISUALIZE
  frameIndex++;

  return err;
}

DoApp::Err DoApp::acquireFaceBoxAndLandmarks() {
  Err err = errNone;
  int numLandmarks = face_ar_engine.getNumLandmarks();
  NvAR_Rect output_bbox;
  std::vector<NvAR_Point2f> facial_landmarks(numLandmarks);

  // get landmarks in  original image resolution coordinate space
  unsigned n = face_ar_engine.acquireFaceBoxAndLandmarks(frame, facial_landmarks.data(), output_bbox, *sender, 0);

  if (n && FLAG_verbose && face_ar_engine.appMode != FaceEngine::mode::faceDetection) {
    printf("Landmarks: [\n");
    for (const auto &pt : facial_landmarks) {
      printf("%7.1f%7.1f\n", pt.x, pt.y);
    }
    printf("]\n");
  }
 
  if (0 == n) return errNoFace;

#ifdef VISUALIZE

  if (drawVisualization) {
    DrawLandmarkPoints(frame, facial_landmarks.data(), numLandmarks);
    if (FLAG_offlineMode) {
      DrawBBoxes(frame, &output_bbox);
    }
  }
#endif  // VISUALIZE
  frameIndex++;

  return err;
}

DoApp::Err DoApp::fitFaceModel() {
  DoApp::Err doErr = errNone;
  nvErr = face_ar_engine.fitFaceModel(frame,*sender);
  
  if (FaceEngine::Err::errNone == nvErr) {
#ifdef VISUALIZE
    frameTimer.pause();
    if (drawVisualization) {
      DrawFaceMesh(frame, face_ar_engine.getFaceMesh());
    }
    frameTimer.resume();
#endif  // VISUALIZE
  } else {
    doErr = errFaceFit;
  }

  return doErr;
}

DoApp::DoApp() {
  // Make sure things are initialized properly
  init_sender((std::string)"127.0.0.1",4242);
  gApp = this;
  drawVisualization = true;
  showFPS = false;
  captureVideo = false;
  captureFrame = false;
  frameTime = 0;
  frameIndex = 0;
  nvErr = FaceEngine::errNone;
  scaleOffsetXY[0] = scaleOffsetXY[2] = 1.f;
  scaleOffsetXY[1] = scaleOffsetXY[3] = 0.f;
}

DoApp::~DoApp() {}

void DoApp::getFPS() {
  const float timeConstant = 16.f;
  frameTimer.stop();
  float t = (float)frameTimer.elapsedTimeFloat();
  if (t < 100.f) {
    if (frameTime)
      frameTime += (t - frameTime) * (1.f / timeConstant);  // 1 pole IIR filter
    else
      frameTime = t;
  } else {            // Ludicrous time interval; reset
    frameTime = 0.f;  // WAKE UP
  }
  frameTimer.start();
}

void DoApp::drawFPS(cv::Mat &img) {
  getFPS();
  if (frameTime && showFPS) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.1f", 1. / frameTime);
    cv::putText(img, buf, cv::Point(img.cols - 80, img.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(255, 255, 255), 1);
  }
}


DoApp::Err DoApp::run() {
  
  DoApp::Err doErr = errNone;
  FaceEngine::Err err = face_ar_engine.initFeatureIOParams();
  if (err != FaceEngine::Err::errNone ) {
    return doAppErr(err);
  }

  while (1) {
    doErr = acquireFrame();
    if (frame.empty() && FLAG_offlineMode) {
      // We have reached the end of the video
      // so return without any error.
      return DoApp::errNone;
    } else if (doErr != DoApp::errNone) {
      return doErr;
    }
    if (face_ar_engine.appMode == FaceEngine::mode::faceDetection) {
      doErr = acquireFaceBox();
    } else if (face_ar_engine.appMode == FaceEngine::mode::landmarkDetection) {
      doErr = acquireFaceBoxAndLandmarks();
    }else if (face_ar_engine.appMode == FaceEngine::mode::faceMeshGeneration) {
      doErr = fitFaceModel();
    }
    if (DoApp::errCancel == doErr || DoApp::errVideo == doErr) return doErr;
    if (!frame.empty() && !FLAG_offlineMode) {
      if (drawVisualization) {
        drawFPS(frame);
      }
      cv::imshow(windowTitle, frame);
    }

    if (!FLAG_offlineMode) {
      int n = cv::waitKey(1);
      if (n >= 0) {
        static const int ESC_KEY = 27;
        if (n == ESC_KEY) break;
        processKey(n);
      }
    }
  }
  return doErr;
}

DoApp::Err DoApp::setProxyWireframe(const char *path) {
  std::ifstream file(path);
  if (!file.is_open()) return errFaceModelInit;
  std::string lineBuf;
  while (getline(file, lineBuf)) {
    std::array<uint16_t, 3> tri;
    if (3 == sscanf(lineBuf.c_str(), "f %hd %hd %hd", &tri[0], &tri[1], &tri[2])) {
      --tri[0]; /* Convert from 1-based to 0-based indices */
      --tri[1];
      --tri[2];
      proxyWireframe.push_back(tri);
    }
  }
  file.close();
  return errNone;
}



/********************************************************************************
 * main
 ********************************************************************************/

int main(int argc, char** argv) {
    DoApp app;
    DoApp::Err doErr;
    

    app.face_ar_engine.setAppMode(FaceEngine::mode(2));

    app.face_ar_engine.setFaceStabilization(FLAG_temporal);

    if (!FLAG_faceModel.empty())
        app.face_ar_engine.setFaceModel(FLAG_faceModel.c_str());

    doErr = app.initCamera(FLAG_camRes.c_str());
    BAIL_IF_ERR(doErr);

    doErr = app.initFaceEngine(FLAG_modelPath.c_str(), FLAG_isNumLandmarks126);
    BAIL_IF_ERR(doErr);

    doErr = app.run();
    BAIL_IF_ERR(doErr);

    

bail:
    if (doErr)
        printf("ERROR: %s\n", app.errorStringFromCode(doErr));
    app.stop();
    return (int)doErr;
}


const char* DoApp::errorStringFromCode(DoApp::Err code) {
    struct LUTEntry {
        Err code;
        const char* str;
    };
    static const LUTEntry lut[] = {
        {errNone, "no error"},
        {errGeneral, "an error has occured"},
        {errRun, "an error has occured while the feature is running"},
        {errInitialization, "Initializing Face Engine failed"},
        {errRead, "an error has occured while reading a file"},
        {errEffect, "an error has occured while creating a feature"},
        {errParameter, "an error has occured while setting a parameter for a feature"},
        {errUnimplemented, "the feature is unimplemented"},
        {errMissing, "missing input parameter"},
        {errVideo, "no video source has been found"},
        {errImageSize, "the image size cannot be accommodated"},
        {errNotFound, "the item cannot be found"},
        {errFaceModelInit, "face model initialization failed"},
        {errGLFWInit, "GLFW initialization failed"},
        {errGLInit, "OpenGL initialization failed"},
        {errRendererInit, "renderer initialization failed"},
        {errGLResource, "an OpenGL resource could not be found"},
        {errGLGeneric, "an otherwise unspecified OpenGL error has occurred"},
        {errFaceFit, "an error has occurred while face fitting"},
        {errNoFace, "no face has been found"},
        {errSDK, "an SDK error has occurred"},
        {errCuda, "a CUDA error has occurred"},
        {errCancel, "the user cancelled"},
        {errCamera, "unable to connect to the camera"},
    };
    for (const LUTEntry* p = lut; p < &lut[sizeof(lut) / sizeof(lut[0])]; ++p)
        if (p->code == code) return p->str;
    static char msg[18];
    snprintf(msg, sizeof(msg), "error #%d", code);
    return msg;
}
