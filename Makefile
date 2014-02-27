# main.c by Hann Woei Ho
# compile with OpenCV
CC = gcc
CXX = g++
CFLAGS = -I/usr/local/include/opencv -I/usr/local/include -c -Wall
LFLAGS = -lm /usr/local/lib/libopencv_calib3d.so /usr/local/lib/libopencv_contrib.so /usr/local/lib/libopencv_core.so /usr/local/lib/libopencv_features2d.so /usr/local/lib/libopencv_flann.so /usr/local/lib/libopencv_gpu.so /usr/local/lib/libopencv_highgui.so /usr/local/lib/libopencv_imgproc.so /usr/local/lib/libopencv_legacy.so /usr/local/lib/libopencv_ml.so /usr/local/lib/libopencv_nonfree.so /usr/local/lib/libopencv_objdetect.so /usr/local/lib/libopencv_ocl.so /usr/local/lib/libopencv_photo.so /usr/local/lib/libopencv_stitching.so /usr/local/lib/libopencv_superres.so /usr/local/lib/libopencv_ts.a /usr/local/lib/libopencv_video.so /usr/local/lib/libopencv_videostab.so /usr/lib/i386-linux-gnu/libXext.so /usr/lib/i386-linux-gnu/libX11.so /usr/lib/i386-linux-gnu/libICE.so /usr/lib/i386-linux-gnu/libSM.so /usr/lib/i386-linux-gnu/libGL.so /usr/lib/i386-linux-gnu/libGLU.so -ltbb -lrt -lpthread -lm -ldl 

all: test

test: fastRosten.o nrutil.o optic_flow_gdc.o main.o
	$(CC) fastRosten.o nrutil.o optic_flow_gdc.o main.o $(LFLAGS) -o test

main.o: main.c
	$(CC) $(CFLAGS) main.c

optic_flow_gdc.o: optic_flow_gdc.c
	$(CC) $(CFLAGS) optic_flow_gdc.c

nrutil.o: nrutil.c
	$(CC) $(CFLAGS) nrutil.c

fastRosten.o: fastRosten.c
	$(CC) $(CFLAGS) fastRosten.c

clean:
	rm -rf *o test
