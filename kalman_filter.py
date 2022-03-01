'''

Odometry or Encoder data from Seekur Mobile robot
field.O_x is Odometry data in x direction/coordinate of the robot
field.O_y is Odometry data in y direction/coordinate of the robot
field.O_t is Odometry data of the orientation or heading of the robot
(For the covariance of the Odometry data you can give a specific number, for example: 0.001.)

Data from Microstrain IMU attached on the robot
field.I_t is IMU data of the orientation or heading of the robot
field.Co_I_t  is the IMU data of the Covariance of the orientation of the robot


Novatel DGPS  data attached on the robot
field.G_x is GPS data in x direction/coordinate of the robot
field.G_y is GPS data in y direction/coordinate of the robot
field.Co_gps_x is GPS data of the Covariance in x direction of the robot
field.Co_gps_y is GPS data of the Covariance in y direction of the robot

'''


from cmath import tan, cos, sin
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
#date_file = 'EKF_DATA_circle.txt'

#data = np.loadtxt(date_file, delimiter=',', skiprows=1, dtype=str)

#print(data[0])

class KalmanFilter():
    def __init__(self):
        self.text_file = 'data/EKF_DATA_circle.txt'
        self.data = 0

        self.time = []

        self.Odom_x = []
        self.Odom_y = []
        self.Odom_theta = []

        self.Gps_x = []
        self.Gps_y = []
        
        self.Gps_Co_x = []
        self.Gps_Co_y = []
        
        self.IMU_heading = []
        self.IMU_Co_heading = []
        
        self.matrix_z = []

        self.velocity = 0.14
        self.dist_wheels = 1
        self.omega = 0
        self.total = 0
        self.delta_t = 0.001

        #storing the data for kalman filter
        self.true = []
        self.X1 = []
        self.X2 = []
        self.X_heading = []



        

        
    def begin_kalman(self):
        print(self.total)
        for i in range(self.total):
            self.matrix_a[0,2] = self.delta_t * cos(self.Odom_theta[i])
            self.matrix_a[1,2] = self.delta_t * sin(self.Odom_theta[i])
            
            self.matrix_R[0,0] = self.Gps_Co_x[i]
            self.matrix_R[1,1] = self.Gps_Co_y[i]
            self.matrix_R[3,3] = self.IMU_Co_heading[i]

            self.matrix_z = np.array(
                [[self.Gps_x[i]], 
                [self.Gps_y[i]],
                [self.velocity],
                [self.IMU_Co_heading[i]],
                [self.omega]]
            )
            #put kalman filter here
            X_f = self.kalman_filter()
            self.X1.append(X_f[0][0])
            self.X2.append(X_f[1][0])
            self.X_heading.append(X_f[3][0])

        plt.plot(self.Odom_x, self.Odom_y,'.', label='ODOM', markersize = 2)
        plt.plot(self.Gps_x, self.Gps_y, '.', label='GPS', markersize = 2)
        plt.plot(self.X1, self.X2, '.', color='red', label='KF',markersize = 2)
        plt.legend()
        plt.show()

    def kalman_filter(self):
        #prediction stage for state vector and co  variance

        X_neg = np.dot(self.matrix_a, self.initial_states)
        #self.initial_states = X_neg
        P = np.dot(np.dot(self.matrix_a, self.matrix_P), np.transpose(self.matrix_a)) + self.matrix_q

        #compuute kalman gain factor
        input = np.dot(np.dot(self.matrix_H, P), np.transpose(self.matrix_H)) + self.matrix_R
        inverse_input = np.linalg.inv(input)
        K = np.dot(np.dot(P, np.transpose(self.matrix_H)), inverse_input)

        #correctionstage base on measurement
        matrix_y = np.dot(self.matrix_H, X_neg)
        X_f = X_neg + np.dot(K, (self.matrix_z - matrix_y))
        self.initial_states = X_f
        self.matrix_P = P - np.dot(np.dot(K, self.matrix_H), P)

        return X_f
        #
        #print(K)

        #print(X)




    def create_dataframe(self):
        #convert the text file into a pandas dataframe

        read_file = pd.read_csv(self.text_file)
        read_file.to_csv(self.text_file + '.csv', index=None)
        

        self.data = read_file

        #print(self.data)

    def get_data(self): 
        self.time = self.data['%time']
        self.Odom_x = self.data['field.O_x']
        self.Odom_y = self.data['field.O_y']
        self.Odom_theta = self.data['field.O_t']

        self.Gps_x = self.data['field.G_x']
        self.Gps_y = self.data['field.G_y']
        
        self.Gps_Co_x = self.data['field.Co_gps_x']
        self.Gps_Co_y = self.data['field.Co_gps_y']
        
        self.IMU_heading = self.data['field.I_t']
        self.IMU_Co_heading = self.data['field.Co_I_t']

        self.omega = self.velocity * tan(self.Odom_theta[1]) / self.dist_wheels
        #print(self.IMU_heading.shape)
        #matching with robot's heading initially
        #print(self.Odom_theta[1])
        self.IMU_heading = self.IMU_heading + (0.32981-0.237156) * np.ones((len(self.IMU_heading), ), dtype=int)
        #print(self.Odom_theta[1])
        self.initial_states = np.array(
            [[self.Odom_x[0]], 
            [self.Odom_y[0]],
            [self.velocity], 
            [self.Odom_theta[0]], 
            [self.omega]]
        )

        self.total = len(self.Odom_x)
        #print(self.initial_states)

    def noise_gps_inc(self):
        noise_mean = 0.5
        noise_std = 0.12
        #print(np.random.randn(len(self.Odom_x), 2))
        gps_noise = noise_std * np.random.randn(len(self.Odom_x), 2) +  noise_mean * np.ones((len(self.Odom_x), 2))
        
        
        for i in range(1000):
            #print(gps_noise[i][0])
            self.Gps_x[i] += gps_noise[i][0]
            self.Gps_y[i] += gps_noise[i][1]


            self.Gps_Co_x[i] += gps_noise[i][0]
            self.Gps_Co_y[i] += gps_noise[i][1]
        for i in range(2000,3000):
            self.Gps_x[i] += gps_noise[i][0]
            self.Gps_y[i] += gps_noise[i][1]


            self.Gps_Co_x[i] += gps_noise[i][0]
            self.Gps_Co_y[i] += gps_noise[i][1]

        
    def noise_gps(self):
        noise_mean = 0.5
        noise_std = 0.12
        #print(np.random.randn(len(self.Odom_x), 2))
        gps_noise = noise_std * np.random.randn(len(self.Odom_x), 2) +  noise_mean * np.ones((len(self.Odom_x), 2))

        self.Gps_x = self.data['field.G_x'] + gps_noise[:,0]
        self.Gps_y = self.data['field.G_y'] + gps_noise[:,1]


        self.Gps_Co_x = self.data['field.Co_gps_x'] + gps_noise[:,0]
        self.Gps_Co_y = self.data['field.Co_gps_y'] + gps_noise[:,1]

    def noise_odom(self):
        noise_mean = 0.5
        noise_std = 0.1
        
        odom_noise = noise_std * np.random.randn(len(self.Odom_x), 2) +  noise_mean * np.ones((len(self.Odom_x), 2))   

        self.Odom_x = self.data['field.O_x'] + odom_noise[:,0]
        self.Odom_y = self.data['field.O_y'] + odom_noise[:,1]

    def noise_imu(self):
        noise_mean = 0.5
        noise_std = 0.1
        
        imu_noise = noise_std * np.random.randn(len(self.Odom_x), 2) +  noise_mean * np.ones((len(self.Odom_x), 2))   

        #for i in range(1000):
         #   self.IMU_Co_heading += imu_noise[i][0]

        #for i in range(2000,3000):
           # self.IMU_Co_heading += imu_noise[i][0]


        self.IMU_Co_heading = self.data['field.Co_I_t'] + imu_noise[:,0]

    def define_matrix(self):
        self.matrix_a = np.array(
            [[1, 0, self.delta_t*cos(self.Odom_theta[0]), 0, 0],
            [0, 1, self.delta_t*sin(self.Odom_theta[0]), 0, 0],
            [0, 0, 1,                                    0, 1],
            [0, 0, 0,                                    1, self.delta_t],
            [0, 0, 0,                                    0, 1]]
        )
        
        self.matrix_q = np.array(
            [[0.0004, 0, 0, 0, 0],
            [0, 0.0004, 0, 0, 0],
            [0, 0, 0.001, 0, 0],
            [0, 0, 0, 0.001, 0],
            [0, 0, 0, 0, 0.001]]
        )

        self.matrix_H = np.array(
            [[1,0,0,0,0],
            [0,1,0,0,0],
            [0,0,1,0,0],
            [0,0,0,1,0],
            [0,0,0,0,1]]
        )

        self.matrix_R = np.array(
            [[.04,  0,   0,   0,   0],
            [0,  .04,  0,   0,   0],
            [0,   0,  .01 , 0,   0],
            [0,   0,   0,   0.01,  0],
            [0,   0,   0,   0,  .01]]
        )

        self.matrix_B = np.array(
            [[1,   0,   0,   0,   0],
            [0,   1,   0 ,  0 ,  0],
            [0,   0,   1,   0,   0],
            [0 ,  0,   0,   1,   0],
            [0,   0,   0,   0,   1]]
        )

        self.matrix_u = np.array([[0],[0],[0],[0],[0]])

        self.matrix_P = np.array(
            [[.001,  0,   0,   0,   0],
            [0,  .001,  0,  0,   0],
            [0,   0,  .001,  0,   0],
            [0,   0,   0,  .001,  0],
            [0,   0,   0,   0,  .001]]
        )





    def plot_data(self):
        fig, ax = plt.subplots(4)
        ax[0].plot(self.time, self.Odom_theta)
        ax[1].plot(self.time, self.data['field.I_t'])
        ax[2].plot(self.Odom_x, self.Odom_y)
        ax[3].plot(self.time, self.IMU_heading)
        plt.show()


if __name__ == '__main__':
    filter = KalmanFilter()
    filter.create_dataframe()
    filter.get_data()
    #filter.noise_gps()
    #filter.noise_gps_inc()
    filter.noise_imu()
    #filter.noise_odom()
    #filter.plot_data()
    filter.define_matrix()

    filter.begin_kalman()
    filter.kalman_filter()