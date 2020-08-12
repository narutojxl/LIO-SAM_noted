# Cloud Info
Header header 

int32[] startRingIndex
int32[] endRingIndex

int32[]  pointColInd # point column index in range image
float32[] pointRange # point range 

int64 imuAvailable
int64 odomAvailable

# Attitude for LOAM initialization
float32 imuRollInit   #imu测量的当前帧laser在全局下的的RPY，imageProjection赋值
float32 imuPitchInit
float32 imuYawInit

# Initial guess from imu pre-integration
float32 initialGuessX  #每一帧laser在map下pose初值, imuPreintegration模块预测发布的。imageProjection赋值
float32 initialGuessY
float32 initialGuessZ
float32 initialGuessRoll
float32 initialGuessPitch
float32 initialGuessYaw

# Point cloud messages
sensor_msgs/PointCloud2 cloud_deskewed  # original cloud deskewed, 全部都转换到了本帧的start点下， imageProjection赋值
sensor_msgs/PointCloud2 cloud_corner    # extracted corner feature  #featureExtraction赋值
sensor_msgs/PointCloud2 cloud_surface   # extracted surface feature #featureExtraction赋值