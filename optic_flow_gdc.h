#ifndef OPTIC
#define OPTIC

int getMaximum(int * Im);
int getMinimum(int * Im);
void getGradientPixelWH(unsigned char *frame_buf, int x, int y, int* dx, int* dy);
void getSimpleGradient(unsigned char* frame_buf, int* DX, int* DY);
void multiplyImages(int* ImA, int* ImB, int* ImC, int width, int height);
void getImageDifference(int* ImA, int* ImB, int* ImC, int width, int height);
int calculateError(int* ImC, int width, int height);
void printIntMatrix(int* Matrix, int width, int height);
void printIntMatrixPart(int* Matrix, int width, int height, int n_cols, int n_rows, int x_center, int y_center);
void smoothGaussian(int* Src, int* Dst);
void getHarris(int* DXX, int* DXY, int* DYY, int* Harris);
int findLocalMaxima(int* Harris, int max_val, int MAX_POINTS, int* p_x, int* p_y, int suppression_distance_squared, int* n_found_points);
void excludeArea(unsigned int* Mask, int x, int y, int suppression_distance_squared);
void thresholdImage(int* Harris, int max_val, int max_factor);
int findCorners(unsigned char *frame_buf, int MAX_POINTS, int *x, int *y, int suppression_distance_squared, int* n_found_points, int mark_points, int imW, int imH);
int findActiveCorners(unsigned char *frame_buf, unsigned int GRID_ROWS, int ONLY_STOPPED, int *x, int *y, int* active, int* n_found_points, int mark_points, int imW, int imH);
void getSubPixel(int* Patch, unsigned char* buf, int center_x, int center_y, int half_window_size, int subpixel_factor);
int calculateG(int* G, int* DX, int* DY, int half_window_size);
void getGradientPatch(int* Patch, int* DX, int* DY, int half_window_size);
int getSumPatch(int* Patch, int size);
void showFlow(unsigned char * frame_buf, int* x, int* y, int* status, int n_found_points, int* new_x, int* new_y, int imgW, int imgH);
int opticFlowLK(unsigned char * new_image_buf, unsigned char * old_image_buf, int* p_x, int* p_y, int n_found_points, int imW, int imH, int* new_x, int* new_y, int* status, int half_window_size, int max_iterations);
void MatMul(float* Mat1, float* Mat2, float* Mat3, int MatW, int MatH);
void MatVVMul(float* MVec, float** Mat, float* Vec, int MatW, int MatH);
void ScaleAdd(float* Mat3, float* Mat1, float Scale, float* Mat2, int MatW, int MatH);
static float PYTHAG(float a, float b);
int dsvd(float **a, int m, int n, float *w, float **v);
void svbksb(float **u, float *w, float **v, int m, int n, float *b, float *x);
void svdSolve(float *x_svd, int **u, int m, int n, int *b);
void fitLinearFlowField(float* pu, float* pv, float* divergence_error, int *x, int *y, int *dx, int *dy, int count, int n_samples, float* min_error_u, float* min_error_v, int n_iterations, float error_threshold, int *n_inlier_minu, int *n_inlier_minv);
void extractInformationFromLinearFlowField(float *divergence, float *mean_tti, float *median_tti, float *d_heading, float *d_pitch, float* pu, float* pv, int imgWidth, int imgHeight, float FPS);
void quick_sort (float *a, int n);
void CvtYUYV2Gray(unsigned char *grayframe, unsigned char *frame, int imW, int imH);
#endif
