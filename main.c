/*
 * main.c
 *
 *  Created on: Feb 12, 2014
 *      Author: h2w
 *      To show video streaming and plot real-time results:
 *      ./test | ./driveGnuPlotStreams.pl 2 50 50 0 80 -0.1 0.1 count divergence 500x300+0+0 500x300+500+0
 */


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
// OpenCV
#include <cv.h>
#include <highgui.h>
// v4l2
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <getopt.h>             /* getopt_long() */

#include <fcntl.h>              /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include <linux/videodev2.h>

// Computer vision
#include "fastRosten.h"
#include "optic_flow_gdc.h"


// Corner Detection
int count = 0;
int max_count = 25;
int error_corner;
int flow_point_size = 0;
#define MAX_COUNT 100	// Maximum number of flow points

unsigned char *frame, *gray_frame, *prev_gray_frame;

IplImage *pyramid = 0;
IplImage *prev_pyramid = 0;
IplImage *swap_temp = 0;
CvPoint2D32f* swap_points;
IplImage* cvGrayImg = 0;
IplImage* cvPrevGrayImg = 0;
CvPoint2D32f* points[3];
int flags = 0;
char* status = 0;

typedef struct flowPoint
{
	double x;
	double y;
	double prev_x;
	double prev_y;
	double dx;
	double dy;
	double new_dx;
	double new_dy;
//	double P[16]; // represents a diagonal 4x4 matrix
//	double Q[16]; // represents a diagonal 4x4 matrix
//	double R[16]; // represents a diagonal 4x4 matrix
//	double K[16]; // represents a diagonal 4x4 matrix
//	int n_observations;
} flowPoint;

flowPoint flow_points[100];

#define CLEAR(x) memset(&(x), 0, sizeof(x))

enum io_method {
        IO_METHOD_READ,
        IO_METHOD_MMAP,
        IO_METHOD_USERPTR,
};

struct buffer {
        void   *start;
        size_t  length;
};

static char            *dev_name;
static enum io_method   io = IO_METHOD_MMAP;
static int              fd = -1;
struct buffer          *buffers;
static unsigned int     n_buffers;
static int              out_buf;
static int              force_format;
static int              frame_count = 70;

static int imW = 640;
static int imH = 480;

static void errno_exit(const char *s)
{
        fprintf(stderr, "%s error %d, %s\n", s, errno, strerror(errno));
        exit(EXIT_FAILURE);
}

static int xioctl(int fh, int request, void *arg)
{
        int r;

        do {
                r = ioctl(fh, request, arg);
        } while (-1 == r && EINTR == errno);

        return r;
}

/* convert from 4:2:2 YUYV interlaced to RGB24 */
/* based on ccvt_yuyv_bgr32() from camstream */
#define SAT(c) \
   if (c & (~255)) { if (c < 0) c = 0; else c = 255; }

static void yuyv_to_rgb24 (int width, int height, unsigned char *src, unsigned char *dst)
{
   unsigned char *s;
   unsigned char *d;
   int l, c;
   int r, g, b, cr, cg, cb, y1, y2;

   l = height;
   s = src;
   d = dst;
   while (l--) {
      c = width >> 1;
      while (c--) {
         y1 = *s++;
         cb = ((*s - 128) * 454) >> 8;
         cg = (*s++ - 128) * 88;
         y2 = *s++;
         cr = ((*s - 128) * 359) >> 8;
         cg = (cg + (*s++ - 128) * 183) >> 8;

         r = y1 + cr;
         b = y1 + cb;
         g = y1 - cg;
         SAT(r);
         SAT(g);
         SAT(b);

     *d++ = b;
     *d++ = g;
     *d++ = r;

         r = y2 + cr;
         b = y2 + cb;
         g = y2 - cg;
         SAT(r);
         SAT(g);
         SAT(b);

     *d++ = b;
     *d++ = g;
     *d++ = r;
      }
   }
}

static void show_image(unsigned char *p, int *x, int *y, int *new_x, int *new_y, int n_found_points)
{
		int i;
		unsigned char *rgb;
		rgb =(unsigned char *) calloc(imW*imH*2,sizeof(unsigned char));
		yuyv_to_rgb24 (imW, imH, p, rgb);
        CvMat cvmat = cvMat(imH, imW, CV_8UC3, rgb);
        IplImage* img =  (IplImage*)&cvmat;

        for(i=0; i<n_found_points; i++)
        {
			int radius = 10;
			cvCircle(img,
					cvPoint((int)(x[i] + 0.5f),(int)(y[i] + 0.5f)),
					radius,
					cvScalar(0,0,255,0),1,8,0);
//			CvPoint p0 = cvPoint( cvRound( x[i] ), cvRound( y[i] ) );
//			CvPoint p1 = cvPoint( cvRound( x[i] ), cvRound( y[i] ) );
//			cvLine( img, p0, p1, CV_RGB(255,0,0), 1, 8, 0);
        }
		cvShowImage("window", img);
//		cvWaitKey(0);
		free(rgb);

}

void setPointsToFlowPoints(struct flowPoint flow_points[])
{
	// set the points array to match the flow points:
	int new_size = (flow_point_size < MAX_COUNT) ? flow_point_size : MAX_COUNT;
	count = new_size;
	int i;
	for(i = 0; i < new_size; i++)
	{
		points[0][i].x = (float)flow_points[i].x;
		points[0][i].y = (float)flow_points[i].y;
	}
}

void findPoints(struct flowPoint flow_points[])
{
	// a) find suitable points in the image
	// b) compare their locations with those of flow_points, only allowing new points if far enough
	// c) update flow_points (and points) to include the new points
	// d) put the flow point into the points-array, which will be used for the flow

	// a)

	// FAST corner:
	int fast_threshold = 40; //20
	xyFAST* pnts_fast;

//	CvtYUYV2Gray(gray_frame, frame, imW, imH); // convert to gray scaled image is a must for FAST corner
	pnts_fast = fast9_detect((const byte*)gray_frame, imW, imH, imW, fast_threshold, &count);

	// transform the points to the format we need (is also done in the other corner finders
	count = (count > MAX_COUNT) ? MAX_COUNT : count;
	int i,j;
	for(i = 0; i < count; i++)
	{
		points[0][i].x = pnts_fast[i].x;
		points[0][i].y = pnts_fast[i].y;
	}

	// b)
	float distance2;
	float min_distance = 10;
	float min_distance2 = min_distance*min_distance;
	int new_point;

	int max_new_points = (count < max_count - flow_point_size) ? count : max_count - flow_point_size; //flow_point_size = [0,25]

	for(i = 0; i < max_new_points; i++)
	{
		new_point = 1;

		for(j = 0; j < flow_point_size; j++)
		{
			// distance squared:
			distance2 = (points[0][i].x - flow_points[j].x)*(points[0][i].x - flow_points[j].x) +
						(points[0][i].y - flow_points[j].y)*(points[0][i].y - flow_points[j].y);
			if(distance2 < min_distance2)
			{
				new_point = 0;
			}
		}

		// c)
		if(new_point)
		{
			// add the flow_points:
			flow_points[flow_point_size].x = points[0][i].x;
			flow_points[flow_point_size].y = points[0][i].y;
			flow_points[flow_point_size].prev_x = points[0][i].x;
			flow_points[flow_point_size].prev_y = points[0][i].y;
			flow_points[flow_point_size].dx = 0;
			flow_points[flow_point_size].dy = 0;
			flow_points[flow_point_size].new_dx = 0;
			flow_points[flow_point_size].new_dy = 0;
			flow_point_size++;
		}
	}
	setPointsToFlowPoints(flow_points);
}

void trackPoints(unsigned char *gray_frame, unsigned char *prev_gray_frame, struct flowPoint flow_points[])
{
	// a) track the points to the new image
	// b) quality checking  for eliminating points (status / match error / tracking the features back and comparing / etc.)
	// c) update the points (immediate update / Kalman update)

	int i;

    CvMat cvGray = cvMat(imH, imW, CV_8U, gray_frame);
    CvMat cvPrevGray = cvMat(imH, imW, CV_8U, prev_gray_frame);
    cvGrayImg =  (IplImage*)&cvGray;
    cvPrevGrayImg =  (IplImage*)&cvPrevGray;

    if(!pyramid)
    {
    	pyramid = cvCreateImage( cvGetSize(cvGrayImg), 8, 1 );
    	prev_pyramid = cvCreateImage( cvGetSize(cvPrevGrayImg), 8, 1 );
//    	char* status = (char*) malloc(MAX_COUNT*sizeof(char));
    	status = (char*) cvAlloc(MAX_COUNT);
//    	float* error = (float*) malloc(MAX_COUNT*sizeof(float));
    }

	if(flow_point_size > MAX_COUNT) printf("PROBLEM PROBLEM PROBLEM - too many points!");

	// a) track the points to the new image
	if( count > 0)
    {
        cvCalcOpticalFlowPyrLK( cvPrevGrayImg, cvGrayImg, prev_pyramid, pyramid,
            points[0], points[1], count, cvSize(5,5), 3, status, 0,	// points[0]: prev_features
            cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10, 0.03), flags );	// points[1]: curr_features
        flags |= CV_LKFLOW_PYR_A_READY;
//        cvCalcOpticalFlowPyrLK( cvPrevGrayImg, cvGrayImg, prev_pyramid, pyramid,
//            points[0], points[1], count, cvSize(5,5), 3, status, error,	// points[0]: prev_features
//            cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10, 0.03), flags );	// points[1]: curr_features
//        flags |= CV_LKFLOW_PYR_A_READY;
	}

	// b) quality checking  for eliminating points (status / match error / tracking the features back and comparing / etc.)
	int remove_point = 0;
	int c;
	for(i = flow_point_size-1; i >= 0; i-- )
    {
        if(!status[i])
		{
			remove_point = 1;
		}

		// error[i] can also be used, etc.

		if(remove_point)
		{
			// we now erase the point if it is not observed in the new image
			// later we may count it as a single miss, allowing for a few misses
			for(c = i; c < flow_point_size-1; c++)
			{
				flow_points[c].x = flow_points[c+1].x;
			}
			flow_point_size--;
		}
		else
		{
			flow_points[i].new_dx = points[1][i].x - points[0][i].x;
			flow_points[i].new_dy = points[1][i].y - points[0][i].y;
		}
	}

	// c) update the points (immediate update / Kalman update)
	count = flow_point_size;

	int KALMAN_UPDATE = 0;

	if(!KALMAN_UPDATE)
	{
		for(i = 0; i < count; i++)
		{
			// immediate update:
			flow_points[i].dx = flow_points[i].new_dx;
			flow_points[i].dy = flow_points[i].new_dy;
			flow_points[i].prev_x = flow_points[i].x;
			flow_points[i].prev_y = flow_points[i].y;
			flow_points[i].x = flow_points[i].x + flow_points[i].dx;
			flow_points[i].y = flow_points[i].y + flow_points[i].dy;
		}
	}
	else
	{

	}

//	free(error);
//	free(status);

	return;
}

static void process_image(unsigned char *p, int size)
{
//        if (out_buf)
//                fwrite(p, size, 1, stdout);

        int *x, *y, *new_x, *new_y, i, error_opticflow, *dx, *dy, *n_inlier_minu, *n_inlier_minv;

        x = (int *) calloc(MAX_COUNT,sizeof(int));
        y = (int *) calloc(MAX_COUNT,sizeof(int));
        new_x = (int *) calloc(MAX_COUNT,sizeof(int));
        new_y = (int *) calloc(MAX_COUNT,sizeof(int));
    	dx = (int *) calloc(MAX_COUNT,sizeof(int));
    	dy = (int *) calloc(MAX_COUNT,sizeof(int));
    	n_inlier_minu = (int *)calloc(1,sizeof(int));
    	n_inlier_minv = (int *)calloc(1,sizeof(int));

    	float *divergence;
    	divergence = (float *) calloc(1,sizeof(float));

        memcpy(frame,p,size);

    	CvtYUYV2Gray(gray_frame, frame, imW, imH); // convert to gray scaled image is a must for FAST corner

		// ***********************************************************************************************************************
		// (1) possibly find new points - keeping possible old ones (normal cv methods / efficient point finding / active corners)
		// ***********************************************************************************************************************

        int ALWAYS_NEW_POINTS = 0;

        if(ALWAYS_NEW_POINTS)
        {
        	// Clear corners
        	memset(flow_points,0,sizeof(flowPoint)*flow_point_size);
        	findPoints(flow_points);
        }
        else
        {
        	int threshold_n_points = 25;
        	if(flow_point_size < threshold_n_points)
        	{
        		findPoints(flow_points);
        	}
        }

		// **********************************************************************************************************************
		// (2) track the points to the new image, possibly using external information (TTC, known lateral / rotational movements)
		// **********************************************************************************************************************

        if(count)
        {
    		trackPoints(gray_frame, prev_gray_frame, flow_points);
        }

		for(i=0; i<flow_point_size; i++)
		{
			x[i] = flow_points[i].x;
			y[i] = flow_points[i].y;
			dx[i] = flow_points[i].dx;
			dy[i] = flow_points[i].dy;
		}

		int start_fitting = 0;

		if(start_fitting == 1)
		{
			// linear fit of the optic flow field
			float error_threshold = 10; // 10
			int n_iterations = 20; // 40
			int count;
			count = flow_point_size;
			int n_samples = (count < 5) ? count : 5;
			float mean_tti, median_tti, d_heading, d_pitch;

			// minimum = 3
	//		if(n_samples < 3)
	//		{
	//			// set dummy values for tti, etc.
	//			mean_tti = 1000.0f / FPS;
	//			median_tti = mean_tti;
	//			d_heading = 0;
	//			d_pitch = 0;
	//			return;
	//		}
			float pu[3], pv[3];

			float divergence_error;
			float min_error_u, min_error_v;
			fitLinearFlowField(pu, pv, &divergence_error, x, y, dx, dy, count, n_samples, &min_error_u, &min_error_v, n_iterations, error_threshold, n_inlier_minu, n_inlier_minv);

			extractInformationFromLinearFlowField(divergence, &mean_tti, &median_tti, &d_heading, &d_pitch, pu, pv, imW, imH, 60);

			printf("0:%d\n1:%f\n",count,divergence[0]);
		}

//		printf("1 char = %d, size of gray = %d, size of prev_gray = %d\n", sizeof(char), sizeof(gray_frame[0]),sizeof(prev_gray_frame[0]));

		memcpy(prev_gray_frame,gray_frame,imH*imW);

		// *********************************************
		// (5) housekeeping to prepare for the next call
		// *********************************************

		// copy pyramid / gray_small / gray into prev ... / .../ _gray_small:
		CV_SWAP( points[0],			points[1],		swap_points );
//		CV_SWAP( cvPrevGrayImg,		cvGrayImg,		swap_temp   );
		CV_SWAP( prev_pyramid,		pyramid,		swap_temp   );

        fflush(stderr);
//        fprintf(stderr, ".");
        fflush(stdout);


        show_image(p, x, y, new_x, new_y, flow_point_size);

        free(x);
        free(y);
        free(new_x);
        free(new_y);
        free(dx);
        free(dy);
        free(n_inlier_minu);
        free(n_inlier_minv);
        free(divergence);

}

static int read_frame(void)
{
        struct v4l2_buffer buf;
        unsigned int i;

        switch (io) {
        case IO_METHOD_READ:
                if (-1 == read(fd, buffers[0].start, buffers[0].length)) {
                        switch (errno) {
                        case EAGAIN:
                                return 0;

                        case EIO:
                                /* Could ignore EIO, see spec. */

                                /* fall through */

                        default:
                                errno_exit("read");
                        }
                }

                process_image(buffers[0].start, buffers[0].length);
                break;

        case IO_METHOD_MMAP:
                CLEAR(buf);

                buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                buf.memory = V4L2_MEMORY_MMAP;

                if (-1 == xioctl(fd, VIDIOC_DQBUF, &buf)) {
                        switch (errno) {
                        case EAGAIN:
                                return 0;

                        case EIO:
                                /* Could ignore EIO, see spec. */

                                /* fall through */

                        default:
                                errno_exit("VIDIOC_DQBUF");
                        }
                }

                assert(buf.index < n_buffers);

                process_image(buffers[buf.index].start, buf.bytesused);




                if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
                        errno_exit("VIDIOC_QBUF");


                break;

        case IO_METHOD_USERPTR:
                CLEAR(buf);

                buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                buf.memory = V4L2_MEMORY_USERPTR;

                if (-1 == xioctl(fd, VIDIOC_DQBUF, &buf)) {
                        switch (errno) {
                        case EAGAIN:
                                return 0;

                        case EIO:
                                /* Could ignore EIO, see spec. */

                                /* fall through */

                        default:
                                errno_exit("VIDIOC_DQBUF");
                        }
                }

                for (i = 0; i < n_buffers; ++i)
                        if (buf.m.userptr == (unsigned long)buffers[i].start
                            && buf.length == buffers[i].length)
                                break;

                assert(i < n_buffers);

                process_image((void *)buf.m.userptr, buf.bytesused);

                if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
                        errno_exit("VIDIOC_QBUF");
                break;
        }

        return 1;
}

static void mainloop(void)
{

	frame = (unsigned char *) calloc(imW*imH*2,sizeof(unsigned char));
	gray_frame = (unsigned char *) calloc(imW*imH,sizeof(unsigned char));
	prev_gray_frame = (unsigned char *) calloc(imW*imH,sizeof(unsigned char));

	int it;
	for ( it=0; it<3; it++)
	{
		points[it] = 0;
	}

	// Allocate feature point buffer (once)
	points[0]		= (CvPoint2D32f*)cvAlloc(MAX_COUNT*sizeof(points[0][0]));
    points[1]		= (CvPoint2D32f*)cvAlloc(MAX_COUNT*sizeof(points[0][0]));
	points[2]		= (CvPoint2D32f*)cvAlloc(MAX_COUNT*sizeof(points[0][0]));

        while (1) {
                for (;;) {
                        fd_set fds;
                        struct timeval tv;
                        int r;

                        FD_ZERO(&fds);
                        FD_SET(fd, &fds);

                        /* Timeout. */
                        tv.tv_sec = 2;
                        tv.tv_usec = 0;

                        r = select(fd + 1, &fds, NULL, NULL, &tv);

                        if (-1 == r) {
                                if (EINTR == errno)
                                        continue;
                                errno_exit("select");
                        }

                        if (0 == r) {
                                fprintf(stderr, "select timeout\n");
                                exit(EXIT_FAILURE);
                        }

                        if (read_frame())
                                break;
                        /* EAGAIN - continue select loop. */
                }

                int k = cvWaitKey(33);

                if (k==27) break;
        }

}

static void stop_capturing(void)
{
        enum v4l2_buf_type type;

        switch (io) {
        case IO_METHOD_READ:
                /* Nothing to do. */
                break;

        case IO_METHOD_MMAP:
        case IO_METHOD_USERPTR:
                type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                if (-1 == xioctl(fd, VIDIOC_STREAMOFF, &type))
                        errno_exit("VIDIOC_STREAMOFF");
                break;
        }
}

static void start_capturing(void)
{
        unsigned int i;
        enum v4l2_buf_type type;

        switch (io) {
        case IO_METHOD_READ:
                /* Nothing to do. */
                break;

        case IO_METHOD_MMAP:
                for (i = 0; i < n_buffers; ++i) {
                        struct v4l2_buffer buf;

                        CLEAR(buf);
                        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                        buf.memory = V4L2_MEMORY_MMAP;
                        buf.index = i;

                        if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
                                errno_exit("VIDIOC_QBUF");
                }
                type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                if (-1 == xioctl(fd, VIDIOC_STREAMON, &type))
                        errno_exit("VIDIOC_STREAMON");
                break;

        case IO_METHOD_USERPTR:
                for (i = 0; i < n_buffers; ++i) {
                        struct v4l2_buffer buf;

                        CLEAR(buf);
                        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                        buf.memory = V4L2_MEMORY_USERPTR;
                        buf.index = i;
                        buf.m.userptr = (unsigned long)buffers[i].start;
                        buf.length = buffers[i].length;

                        if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
                                errno_exit("VIDIOC_QBUF");
                }
                type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                if (-1 == xioctl(fd, VIDIOC_STREAMON, &type))
                        errno_exit("VIDIOC_STREAMON");
                break;
        }
}

static void uninit_device(void)
{
        unsigned int i;

        switch (io) {
        case IO_METHOD_READ:
                free(buffers[0].start);
                break;

        case IO_METHOD_MMAP:
                for (i = 0; i < n_buffers; ++i)
                        if (-1 == munmap(buffers[i].start, buffers[i].length))
                                errno_exit("munmap");
                break;

        case IO_METHOD_USERPTR:
                for (i = 0; i < n_buffers; ++i)
                        free(buffers[i].start);
                break;
        }

        free(buffers);
}

static void init_read(unsigned int buffer_size)
{
        buffers = calloc(1, sizeof(*buffers));

        if (!buffers) {
                fprintf(stderr, "Out of memory\n");
                exit(EXIT_FAILURE);
        }

        buffers[0].length = buffer_size;
        buffers[0].start = malloc(buffer_size);

        if (!buffers[0].start) {
                fprintf(stderr, "Out of memory\n");
                exit(EXIT_FAILURE);
        }
}

static void init_mmap(void)
{
        struct v4l2_requestbuffers req;

        CLEAR(req);

        req.count = 4;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;

        if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req)) {
                if (EINVAL == errno) {
                        fprintf(stderr, "%s does not support "
                                 "memory mapping\n", dev_name);
                        exit(EXIT_FAILURE);
                } else {
                        errno_exit("VIDIOC_REQBUFS");
                }
        }

        if (req.count < 2) {
                fprintf(stderr, "Insufficient buffer memory on %s\n",
                         dev_name);
                exit(EXIT_FAILURE);
        }

        buffers = calloc(req.count, sizeof(*buffers));

        if (!buffers) {
                fprintf(stderr, "Out of memory\n");
                exit(EXIT_FAILURE);
        }

        for (n_buffers = 0; n_buffers < req.count; ++n_buffers) {
                struct v4l2_buffer buf;

                CLEAR(buf);

                buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                buf.memory      = V4L2_MEMORY_MMAP;
                buf.index       = n_buffers;

                if (-1 == xioctl(fd, VIDIOC_QUERYBUF, &buf))
                        errno_exit("VIDIOC_QUERYBUF");

                buffers[n_buffers].length = buf.length;
                buffers[n_buffers].start =
                        mmap(NULL /* start anywhere */,
                              buf.length,
                              PROT_READ | PROT_WRITE /* required */,
                              MAP_SHARED /* recommended */,
                              fd, buf.m.offset);

                if (MAP_FAILED == buffers[n_buffers].start)
                        errno_exit("mmap");
        }
}

static void init_userp(unsigned int buffer_size)
{
        struct v4l2_requestbuffers req;

        CLEAR(req);

        req.count  = 4;
        req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_USERPTR;

        if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req)) {
                if (EINVAL == errno) {
                        fprintf(stderr, "%s does not support "
                                 "user pointer i/o\n", dev_name);
                        exit(EXIT_FAILURE);
                } else {
                        errno_exit("VIDIOC_REQBUFS");
                }
        }

        buffers = calloc(4, sizeof(*buffers));

        if (!buffers) {
                fprintf(stderr, "Out of memory\n");
                exit(EXIT_FAILURE);
        }

        for (n_buffers = 0; n_buffers < 4; ++n_buffers) {
                buffers[n_buffers].length = buffer_size;
                buffers[n_buffers].start = malloc(buffer_size);

                if (!buffers[n_buffers].start) {
                        fprintf(stderr, "Out of memory\n");
                        exit(EXIT_FAILURE);
                }
        }
}

static void init_device(void)
{
        struct v4l2_capability cap;
        struct v4l2_cropcap cropcap;
        struct v4l2_crop crop;
        struct v4l2_format fmt;
        unsigned int min;

        if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &cap)) {
                if (EINVAL == errno) {
                        fprintf(stderr, "%s is no V4L2 device\n",
                                 dev_name);
                        exit(EXIT_FAILURE);
                } else {
                        errno_exit("VIDIOC_QUERYCAP");
                }
        }

        if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
                fprintf(stderr, "%s is no video capture device\n",
                         dev_name);
                exit(EXIT_FAILURE);
        }

        switch (io) {
        case IO_METHOD_READ:
                if (!(cap.capabilities & V4L2_CAP_READWRITE)) {
                        fprintf(stderr, "%s does not support read i/o\n",
                                 dev_name);
                        exit(EXIT_FAILURE);
                }
                break;

        case IO_METHOD_MMAP:
        case IO_METHOD_USERPTR:
                if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
                        fprintf(stderr, "%s does not support streaming i/o\n",
                                 dev_name);
                        exit(EXIT_FAILURE);
                }
                break;
        }


        /* Select video input, video standard and tune here. */


        CLEAR(cropcap);

        cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        if (0 == xioctl(fd, VIDIOC_CROPCAP, &cropcap)) {
                crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                crop.c = cropcap.defrect; /* reset to default */

                if (-1 == xioctl(fd, VIDIOC_S_CROP, &crop)) {
                        switch (errno) {
                        case EINVAL:
                                /* Cropping not supported. */
                                break;
                        default:
                                /* Errors ignored. */
                                break;
                        }
                }
        } else {
                /* Errors ignored. */
        }


        CLEAR(fmt);

        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (force_format) {
                fmt.fmt.pix.width       = imW;
                fmt.fmt.pix.height      = imH;
                fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
//                fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
//                fmt.fmt.pix.field       = V4L2_FIELD_INTERLACED;
                fmt.fmt.pix.field       = V4L2_FIELD_NONE;

                if (-1 == xioctl(fd, VIDIOC_S_FMT, &fmt))
                        errno_exit("VIDIOC_S_FMT");

                /* Note VIDIOC_S_FMT may change width and height. */
        } else {
                /* Preserve original settings as set by v4l2-ctl for example */
                if (-1 == xioctl(fd, VIDIOC_G_FMT, &fmt))
                        errno_exit("VIDIOC_G_FMT");
        }
        // h2w
        char fourcc[5] = {0};
        strncpy(fourcc, (char *)&fmt.fmt.pix.pixelformat, 4);
/*        printf( "Selected Camera Mode:\n"
                "  Width: %d\n"
                "  Height: %d\n"
                "  PixFmt: %s\n"
                "  Field: %d\n",
                fmt.fmt.pix.width,
                fmt.fmt.pix.height,
                fourcc,
                fmt.fmt.pix.field);*/

        /* Buggy driver paranoia. */
        min = fmt.fmt.pix.width * 2;
        if (fmt.fmt.pix.bytesperline < min)
                fmt.fmt.pix.bytesperline = min;
        min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
        if (fmt.fmt.pix.sizeimage < min)
                fmt.fmt.pix.sizeimage = min;

        switch (io) {
        case IO_METHOD_READ:
                init_read(fmt.fmt.pix.sizeimage);
                break;

        case IO_METHOD_MMAP:
                init_mmap();
                break;

        case IO_METHOD_USERPTR:
                init_userp(fmt.fmt.pix.sizeimage);
                break;
        }
}

static void close_device(void)
{
        if (-1 == close(fd))
                errno_exit("close");

        fd = -1;
}

static void open_device(void)
{
        struct stat st;

        if (-1 == stat(dev_name, &st)) {
                fprintf(stderr, "Cannot identify '%s': %d, %s\n",
                         dev_name, errno, strerror(errno));
                exit(EXIT_FAILURE);
        }

        if (!S_ISCHR(st.st_mode)) {
                fprintf(stderr, "%s is no device\n", dev_name);
                exit(EXIT_FAILURE);
        }

        fd = open(dev_name, O_RDWR /* required */ | O_NONBLOCK, 0);

        if (-1 == fd) {
                fprintf(stderr, "Cannot open '%s': %d, %s\n",
                         dev_name, errno, strerror(errno));
                exit(EXIT_FAILURE);
        }
}

static void usage(FILE *fp, int argc, char **argv)
{
        fprintf(fp,
                 "Usage: %s [options]\n\n"
                 "Version 1.3\n"
                 "Options:\n"
                 "-d | --device name   Video device name [%s]\n"
                 "-h | --help          Print this message\n"
                 "-m | --mmap          Use memory mapped buffers [default]\n"
                 "-r | --read          Use read() calls\n"
                 "-u | --userp         Use application allocated buffers\n"
                 "-o | --output        Outputs stream to stdout\n"
                 "-f | --format        Force format to 640x480 YUYV\n"
                 "-c | --count         Number of frames to grab [%i]\n"
                 "",
                 argv[0], dev_name, frame_count);
}

static const char short_options[] = "d:hmruofc:";

static const struct option
long_options[] = {
        { "device", required_argument, NULL, 'd' },
        { "help",   no_argument,       NULL, 'h' },
        { "mmap",   no_argument,       NULL, 'm' },
        { "read",   no_argument,       NULL, 'r' },
        { "userp",  no_argument,       NULL, 'u' },
        { "output", no_argument,       NULL, 'o' },
        { "format", no_argument,       NULL, 'f' },
        { "count",  required_argument, NULL, 'c' },
        { 0, 0, 0, 0 }
};

int main(int argc, char **argv)
{
	// video capturing options: 1. OpenCV 2. v4l2
	int videoOption = 2;

	if(videoOption == 1) // OpenCV
	{
		//Initializing capture from a camera:
		CvCapture* capture = cvCaptureFromCAM(0); // capture from video device #0

		//Initializing an image
		IplImage* img = 0;

		// create a window
		cvNamedWindow("video", CV_WINDOW_AUTOSIZE);

		while(1)
		{
			//Capturing a frame:
			img = cvQueryFrame( capture );
			if( !img ) break;

			// show the image
			cvShowImage("video", img );

			// wait for a key
			int key=cvWaitKey(10);
			if(key==27) break;
		}

		//Releasing the capture source:
		cvReleaseCapture(&capture);
	}
	else if(videoOption == 2) // v4l2
	{
        dev_name = "/dev/video0";
		cvNamedWindow("window",CV_WINDOW_AUTOSIZE);
        for (;;) {
                int idx;
                int c;
                force_format = 1;
                c = getopt_long(argc, argv,
                                short_options, long_options, &idx);

                if (-1 == c)
                        break;

                switch (c) {
                case 0: /* getopt_long() flag */
                        break;

                case 'd':
                        dev_name = optarg;
                        break;

                case 'h':
                        usage(stdout, argc, argv);
                        exit(EXIT_SUCCESS);

                case 'm':
                        io = IO_METHOD_MMAP;
                        break;

                case 'r':
                        io = IO_METHOD_READ;
                        break;

                case 'u':
                        io = IO_METHOD_USERPTR;
                        break;

                case 'o':
                        out_buf++;
                        break;

                case 'f':
                        force_format++;
                        break;

                case 'c':
                        errno = 0;
                        frame_count = strtol(optarg, NULL, 0);
                        if (errno)
                                errno_exit(optarg);
                        break;

                default:
                        usage(stderr, argc, argv);
                        exit(EXIT_FAILURE);
                }
        }
        open_device();
        init_device();
        start_capturing();
        mainloop();
        stop_capturing();
        uninit_device();
        close_device();
        fprintf(stderr, "\n");

        cvDestroyWindow("window");


	}
	else
	{
		printf("select video capturing options");
	}


	return 0;
}
