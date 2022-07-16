
/**
 * @file
 * video encoding with libavcodec API example
 *
 * @example encode_video.c
 */
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/opencv.hpp>

extern "C" {
   
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

typedef struct config_t {
    std::string outfile_prefix;
    std::string extension;
}config_t;

static void encode_splited(AVCodecContext *enc_ctx, AVFrame *frame, AVPacket *pkt,
                   config_t *config)
{
    int ret;
    /* send the frame to the encoder */
    if (frame)
        printf("Send frame %" PRId64 "\n", frame->pts);
    ret = avcodec_send_frame(enc_ctx, frame);
    if (ret < 0) {
        fprintf(stderr, "Error sending a frame for encoding\n");
        exit(1);
    }
    int cnt=0;
    while (ret >= 0) {
        
        ret = avcodec_receive_packet(enc_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return;
        else if (ret < 0) {
            fprintf(stderr, "Error during encoding\n");
            exit(1);
        }
        std::cout << "Write packet " << pkt->pts << " Size: " << pkt->size << std::endl;;

        std::string filename;
        filename = config->outfile_prefix + std::to_string(cnt) + config->extension;
        FILE *f;
        f = fopen(filename.c_str(), "wb");
        if (!f) {
            std::cerr << "Could not open: " << filename << std::endl;
            exit(1);
        }

        fwrite(pkt->data, 1, pkt->size, f);
        fclose(f);
        av_packet_unref(pkt);
        cnt++;
    }
}

AVFrame *cvmatToAvframe(cv::Mat *image, AVFrame *frame) {
  int width = image->cols;
  int height = image->rows;
  int cvLinesizes[1];
  cvLinesizes[0] = image->step1();
  if (frame == NULL) {
    frame = av_frame_alloc();
    av_image_alloc(frame->data, frame->linesize, width, height,
                   AVPixelFormat::AV_PIX_FMT_YUV420P, 1);
  }
  SwsContext *conversion = sws_getContext(
      width, height, AVPixelFormat::AV_PIX_FMT_BGR24, width, height,
      (AVPixelFormat)frame->format, SWS_FAST_BILINEAR, NULL, NULL, NULL);
  sws_scale(conversion, &image->data, cvLinesizes, 0, height, frame->data,
            frame->linesize);
  sws_freeContext(conversion);
  return frame;
}

static void encode(AVCodecContext *enc_ctx, AVFrame *frame, AVPacket *pkt,
                   FILE *outfile)
{
    int ret;
    
    /* send the frame to the encoder */
    if (frame)
        printf("Send frame %" PRId64 "\n", frame->pts);
    ret = avcodec_send_frame(enc_ctx, frame);
    if (ret < 0) {
        fprintf(stderr, "Error sending a frame for encoding\n");
        exit(1);
    }
    while (ret >= 0) {
        ret = avcodec_receive_packet(enc_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return;
        else if (ret < 0) {
            fprintf(stderr, "Error during encoding\n");
            exit(1);
        }
        printf("Write packet %3" PRId64 " %5d\n", pkt->pts, pkt->size);
        fwrite(pkt->data, 1, pkt->size, outfile);
        av_packet_unref(pkt);
    }
}


int main(int argc, char *argv[])
{

    const char *codec_name;
    const char *bit_rate_arg;

    config_t config;       

    const AVCodec *codec;
    AVCodecContext *c= NULL;
    int i, ret, x, y;
    FILE *f;
    AVFrame *frame;
    AVPacket *pkt;

    
    if (argc <= 2) {
        fprintf(stderr, "Usage: %s <codec name> <bit rate>\n", argv[0]);
        exit(0);
    }
    codec_name = argv[1];
    bit_rate_arg = argv[2];
    
    /* find the mpeg1video encoder */
    codec = avcodec_find_encoder_by_name(codec_name);
    if (!codec) {
        fprintf(stderr, "Codec '%s' not found\n", codec_name);
        exit(1);
    }
    c = avcodec_alloc_context3(codec);
    if (!c) {
        fprintf(stderr, "Could not allocate video codec context\n");
        exit(1);
    }
    pkt = av_packet_alloc();
    if (!pkt)
        exit(1);
    /* put sample parameters */
    c->bit_rate = atoi(bit_rate_arg);
    /* resolution must be a multiple of two */
    c->width = 240;
    c->height = 240;
    /* frames per second */
    c->time_base = (AVRational){1, 30};
    //c->framerate = (AVRational){2, 1};
    
    /* emit one intra frame every ten frames
     * check frame pict_type before passing frame
     * to encoder, if frame->pict_type is AV_PICTURE_TYPE_I
     * then gop_size is ignored and the output of encoder
     * will always be I frame irrespective to gop_size
     */
    //c->gop_size = 10;
    //c->max_b_frames = 1;
    c->pix_fmt = AV_PIX_FMT_YUV444P;

    if (codec->id == AV_CODEC_ID_H264){
        
        config.extension = ".h264"; 
    }
    else if (codec->id == AV_CODEC_ID_H265)
        config.extension = ".h265"; 
    else
        config.extension = ".mp4"; 
    
    av_opt_set(c->priv_data, "preset", "veryfast", 0);
    // f = fopen(filename, "wb");
    // if (!f) {
    //     fprintf(stderr, "Could not open %s\n", filename);
    //     exit(1);
    // }
    config.outfile_prefix = "frame";
    /* open it */
    ret = avcodec_open2(c, codec, NULL);
    if (ret < 0) {
            std::cerr << "Could not open codec "; // << av_err2str(ret) << std::endl;
        exit(1);
    }
    
    frame = av_frame_alloc();
    if (!frame) {
        fprintf(stderr, "Could not allocate video frame\n");
        exit(1);
    }
    frame->format = c->pix_fmt;
    frame->width  = c->width;
    frame->height = c->height;
    ret = av_frame_get_buffer(frame, 32);
    if (ret < 0) {
        fprintf(stderr, "Could not allocate the video frame data\n");
        exit(1);
    }
    /* encode 1 second of video */
    for (i = 0; i < 2; i++) {
        fflush(stdout);
        
        
        /* prepare a dummy image */
        /* Y */
        std::string main_path ="/mnt/WindowsDev/DataSets/Beauty_1920x1080_120fps_420_8bit_YUV_RAW/im";
        std::string img_path = main_path + std::to_string(i+1) + ".png";
        
        cv::Mat img = cv::imread(img_path);
        if (img.empty()) { 
            std::cerr << "Could not open file " << img_path << std::endl; return (1);
        }
        cv::Mat cropedImage = img(cv::Rect(0,0,240,240));
        // cv::cvtColor(cropedImage, cropedImage, cv::COLOR_RGB2YUV);
    
        frame = cvmatToAvframe(&cropedImage, frame);
        /* make sure the frame data is writable */
        ret = av_frame_make_writable(frame);
        if (ret < 0)
            exit(1);

        // cv::Mat channel[3];
        // cv::split(cropedImage, channel);
        // cv::imwrite("cropped_yuv.png", cropedImage);
        
        // // frame->data[0] = channel[0].data;
        // frame->data[1] = channel[1].data;
        // frame->data[2] = channel[1].data;
        // for (y = 0; y < c->height; y++) {
        //     for (x = 0; x < c->width; x++) {
        //         frame->data[0][y * frame->linesize[0] + x] = x + y + i * 3;
        //     }
        // }
        /* Cb and Cr */
        // for (y = 0; y < c->height/2; y++) {
        //     for (x = 0; x < c->width/2; x++) {
        //         frame->data[1][y * frame->linesize[1] + x] = 128 + y + i * 2;
        //         frame->data[2][y * frame->linesize[2] + x] = 64 + x + i * 5;
        //     }
        // }
        frame->pts = i;
        /* encode the image */
        
        encode_splited(c, frame, pkt, &config);
        
        
    }
    
    /* flush the encoder */
   encode_splited(c, NULL, pkt, &config);
    /* add sequence end code to have a real MPEG file */
    // uint8_t endcode[] = { 0, 0, 1, 0xb7 };
    // fwrite(endcode, 1, sizeof(endcode), f);
    // fclose(f);
    avcodec_free_context(&c);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    return 0;
}