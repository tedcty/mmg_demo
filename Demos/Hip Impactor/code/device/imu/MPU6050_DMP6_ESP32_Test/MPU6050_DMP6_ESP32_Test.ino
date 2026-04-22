/* ==============================================
  For ESP32

  MPU6050
      SCL D22
      SDA D21
      INT D35
      INT D34
  LSM6DS3 I2C
      -SDA -> GPIO21
      -SCL -> GPIO22
  SPI IMU



  ===============================================
*/

#include "BluetoothSerial.h"
#include "I2Cdev.h"
#include "MPU6050_6Axis_MotionApps20.h"
#include <LSM6.h>
#include "SparkFunLSM6DS3.h"
#include "SPI.h"
#include <Wire.h>
//#include <SPI.h>

#define pwm_pin 12

//#include "MPU6050.h" // not necessary if using MotionApps include file

// Arduino Wire library is required if I2Cdev I2CDEV_ARDUINO_WIRE implementation
// is used in I2Cdev.h
#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
#include "Wire.h"
#endif

// class default I2C address is 0x68
// specific I2C addresses may be passed as a parameter here
// AD0 low = 0x68 (default for SparkFun breakout and InvenSense evaluation board)
// AD0 high = 0x69
MPU6050 mpu;
MPU6050 mpu2(0x69); // <-- use for AD0 high
//LSM6DS3 mpu3; //Default constructor is I2C, addr 0x6B

LSM6 imu3;

char report[80];

int16_t ax, ay, az;
int16_t gx, gy, gz;


#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run `make menuconfig` to and enable it
#endif

BluetoothSerial SerialBT;

#define INTERRUPT_PIN01 34  // use pin 2 on Arduino Uno & most boards
//#define INTERRUPT_PIN01 35  // use pin 2 on Arduino Uno & most boards



/* =========================================================================
   NOTE: In addition to connection 3.3v, GND, SDA, and SCL, this sketch
   depends on the MPU-6050's INT pin being connected to the Arduino's
   external interrupt #0 pin. On the Arduino Uno and Mega 2560, this is
   digital I/O pin 2.
   ========================================================================= */

/* =========================================================================
   NOTE: Arduino v1.0.1 with the Leonardo board generates a compile error
   when using Serial.write(buf, len). The Teapot output uses this method.
   The solution requires a modification to the Arduino USBAPI.h file, which
   is fortunately simple, but annoying. This will be fixed in the next IDE
   release. For more info, see these links:

   http://arduino.cc/forum/index.php/topic,109987.0.html
   http://code.google.com/p/arduino/issues/detail?id=958
   ========================================================================= */

//// uncomment "OUTPUT_READABLE_YAWPITCHROLL" if you want to see the yaw/
//// pitch/roll angles (in degrees) calculated from the quaternions coming
//// from the FIFO. Note this also requires gravity vector calculations.
//// Also note that yaw/pitch/roll angles suffer from gimbal lock (for
//// more info, see: http://en.wikipedia.org/wiki/Gimbal_lock)
//#define OUTPUT_READABLE_YAWPITCHROLL
//
//// uncomment "OUTPUT_READABLE_REALACCEL" if you want to see acceleration
//// components with gravity removed. This acceleration reference frame is
//// not compensated for orientation, so +X is always +X according to the
//// sensor, just without the effects of gravity. If you want acceleration
//// compensated for orientation, us OUTPUT_READABLE_WORLDACCEL instead.
//#define OUTPUT_READABLE_REALACCEL

#define LED_PIN 2 // (Arduino is 13, Teensy is 11, Teensy++ is 6)
bool blinkState = false;

// MPU control/status vars
bool dmpReady = false;  // set true if DMP init was successful
uint8_t mpuIntStatus;   // holds actual interrupt status byte from MPU
uint8_t devStatus;      // return status after each device operation (0 = success, !0 = error)
uint16_t packetSize;    // expected DMP packet size (default is 42 bytes)
uint16_t fifoCount;     // count of all bytes currently in FIFO
uint8_t fifoBuffer[1024]; // FIFO storage buffer

// orientation/motion vars
Quaternion q;           // [w, x, y, z]         quaternion container
VectorInt16 aa;         // [x, y, z]            accel sensor measurements
VectorInt16 aaReal;     // [x, y, z]            gravity-free accel sensor measurements
//VectorInt16 aaWorld;    // [x, y, z]            world-frame accel sensor measurements
VectorFloat gravity;    // [x, y, z]            gravity vector
//float euler[3];         // [psi, theta, phi]    Euler angle container
float ypr[3];           // [yaw, pitch, roll]   yaw/pitch/roll container and gravity vector

//*********
//// packet structure for InvenSense teapot demo
//uint8_t teapotPacket[14] = { '$', 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0x00, 0x00, '\r', '\n' };



// ================================================================
// ===               INTERRUPT DETECTION ROUTINE                ===
// ================================================================

volatile bool mpuInterrupt = false;     // indicates whether MPU interrupt pin has gone high
void dmpDataReady() {
  mpuInterrupt = true;
}

// ================================================================
// ===                      INITIAL SETUP                       ===
// ================================================================
//  Times
unsigned long startTime = 0;
unsigned long previous = 0;

void setup() {

  pinMode(pwm_pin, OUTPUT);
  digitalWrite(pwm_pin, HIGH);

  // join I2C bus (I2Cdev library doesn't do this automatically)
#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
  Wire.begin();
  //  TWBR = 48; // 400kHz I2C clock (200kHz if CPU is 8MHz)
#elif I2CDEV_IMPLEMENTATION == I2CDEV_BUILTIN_FASTWIRE
  Fastwire::setup(400, true);
#endif

  // initialize serial communication
  // (115200 chosen because it is required forg Teapot Demo output, but it's
  // really up to you depending on your project)
  Serial.begin(115200);
  SerialBT.begin("Orthosens");

  while (!Serial); // wait for Leonardo enumeration, others continue immediately

  // NOTE: 8MHz or slower host processors, like the Teensy @ 3.3v or Ardunio
  // Pro Mini running at 3.3v, cannot handle this baud rate reliably due to
  // the baud timing being too misaligned with processor ticks. You must use
  // 38400 or slower in these cases, or use some kind of external separate
  // crystal solution for the UART timer.

  // initialize device
  Serial.println(F("Initializing I2C devices..."));
  SerialBT.println(F("Initializing I2C devices..."));

  mpu.initialize();
  mpu2.initialize();
  //  mpu3.begin();
  pinMode(INTERRUPT_PIN01, INPUT);
  // verify connection
  Serial.println(F("Testing device connections..."));
  Serial.println(mpu.testConnection() ? F("MPU6050_1 connection successful") : F("MPU6050 connection failed"));
  Serial.println(mpu2.testConnection() ? "MPU6050_2 connection successful" : "MPU6050 connection failed");

  SerialBT.println(F("Testing device connections..."));
  SerialBT.println(mpu.testConnection() ? F("MPU6050_1 connection successful") : F("MPU6050 connection failed"));
  SerialBT.println(mpu2.testConnection() ? "MPU6050_2 connection successful" : "MPU6050 connection failed");

  if (!imu3.init())
  {
    Serial.println("Failed to detect and initialize IMU3!");
    SerialBT.println("Failed to detect and initialize IMU3!");
    while (1);
  }
  imu3.enableDefault();

  // wait for ready
  //    Serial.println(F("\nSend any character to begin DMP programming and demo: "));
  //      while (Serial.available() && Serial.read()); // empty buffer
  //      while (!Serial.available());                 // wait for data
  //      while (Serial.available() && Serial.read()); // empty buffer again

  // load and configure the DMP
  Serial.println(F("Initializing DMP..."));
  SerialBT.println(F("Initializing DMP..."));
  devStatus = mpu.dmpInitialize();

  // supply your own gyro offsets here, scaled for min sensitivity
  //  mpu.setXGyroOffset(220);
  //  mpu.setYGyroOffset(76);
  //  mpu.setZGyroOffset(-85);
  //  mpu.setZAccelOffset(1788); // 1688 factory default for my test chip

  mpu.CalibrateAccel(6);
  mpu.CalibrateGyro(6);
  mpu2.CalibrateAccel(6);
  mpu2.CalibrateGyro(6);
  //  mpu3.CalibrateAccel(6);
  //  mpu3.CalibrateGyro(6);

  // make sure it worked (returns 0 if so)
  if (devStatus == 0) {
    // turn on the DMP, now that it's ready
    Serial.println(F("Enabling DMP..."));
    SerialBT.println(F("Enabling DMP..."));
    mpu.setDMPEnabled(true);

    // enable Arduino interrupt detection
    Serial.println(F("Enabling interrupt detection (Arduino external interrupt 0)..."));
    Serial.print(digitalPinToInterrupt(INTERRUPT_PIN01));
    Serial.println(F(")..."));


    SerialBT.println(F("Enabling interrupt detection (Arduino external interrupt 0)..."));
    SerialBT.print(digitalPinToInterrupt(INTERRUPT_PIN01));
    SerialBT.println(F(")..."));
    attachInterrupt(digitalPinToInterrupt(INTERRUPT_PIN01), dmpDataReady, RISING);

    //    attachInterrupt(0, dmpDataReady, RISING);
    mpuIntStatus = mpu.getIntStatus();

    // set our DMP Ready flag so the main loop() function knows it's okay to use it
    Serial.println(F("DMP ready! Waiting for first interrupt..."));
    SerialBT.println(F("DMP ready! Waiting for first interrupt..."));
    dmpReady = true;

    // get expected DMP packet size for later comparison
    packetSize = mpu.dmpGetFIFOPacketSize();
  } else {
    // ERROR!
    // 1 = initial memory load failed
    // 2 = DMP configuration updates failed
    // (if it's going to break, usually the code will be 1)
    Serial.print(F("DMP Initialization failed (code "));
    Serial.print(devStatus);
    Serial.println(F(")"));
    SerialBT.print(F("DMP Initialization failed (code "));
    SerialBT.print(devStatus);
    SerialBT.println(F(")"));
  }

  // configure LED for output
  pinMode(LED_PIN, OUTPUT);

  startTime = micros();
  previous = startTime;
}
int ticket = 1;
void printOnlyOnce (String message) {
  if (ticket == 1) {
    Serial.println(message);
    ticket = 0 ;
  } else {
    return;
  }
}
// ================================================================
// ===                    MAIN PROGRAM LOOP                     ===
// ================================================================

void loop() {

  unsigned long timeNow = micros();
  float t = ((timeNow - startTime) / 1000.0) / 1000.0;
  float hz = 1000.0 / ((timeNow - previous) / 1000.0);

  // read raw accel/gyro measurements from device
  mpu2.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  imu3.read();

  
  if (!dmpReady) {
    printOnlyOnce("MAIN LOOP: DMP disabled");
    return;
  } else {

    //testing overflow
    if (fifoCount == 1024) {
      mpu.resetFIFO();
      Serial.println("FIFO overflow");
    } else {


      //wait for enough avaliable data length
      while (fifoCount < packetSize) {
        //waiting until get enough
        fifoCount = mpu.getFIFOCount();

      }

      //read this packet from FIFO buffer
      mpu.getFIFOBytes(fifoBuffer, packetSize);
      //track FIFO count here is more then one packeage avalible
      mpu.resetFIFO();

      //reset fifo count
      fifoCount -= packetSize ;
//          Serial.println(fifoCount);

      if (fifoCount > 2) {
        ////// clear fifo buffer
      }
      //****************************************************


      // reset interrupt flag and get INT_STATUS byte
      mpuInterrupt = false;
      mpuIntStatus = mpu.getIntStatus();

      // get current FIFO count
      fifoCount = mpu.getFIFOCount();

      // check for overflow (this should never happen unless our code is too inefficient)
      if ((mpuIntStatus & 0x10) || fifoCount == 16384) { // I changed 1024 -> 16384
        // reset so we can continue cleanly
        mpu.resetFIFO();
        Serial.println(F("FIFO overflow!"));
        SerialBT.println(F("FIFO overflow!"));

        // otherwise, check for DMP data ready interrupt (this should happen frequently)
      } else if (mpuIntStatus & 0x02) {
        // wait for correct available data length, should be a VERY short wait
        while (fifoCount < packetSize) fifoCount = mpu.getFIFOCount();

        // read a packet from FIFO
        mpu.getFIFOBytes(fifoBuffer, packetSize);

        // track FIFO count here in case there is > 1 packet available
        // (this lets us immediately read more without waiting for an interrupt)
        fifoCount -= packetSize;
        snprintf(report, sizeof(report), "A %2d %2d %2d G %2d %2d %2d",
                 imu3.a.x, imu3.a.y, imu3.a.z,
                 imu3.g.x, imu3.g.y, imu3.g.z);

        //#ifdef OUTPUT_READABLE_YAWPITCHROLL
        // display Euler angles in degrees
        mpu.dmpGetQuaternion(&q, fifoBuffer);
        mpu.dmpGetGravity(&gravity, &q);
        mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);

        Serial.print(t);
        Serial.print(", ");
        Serial.print(hz);
        Serial.print("HZ, ");

        Serial.print("ypr ");
        Serial.print(ypr[0] * 180 / M_PI);
        Serial.print(",");
        Serial.print(ypr[1] * 180 / M_PI);
        Serial.print(",");
        Serial.print(ypr[2] * 180 / M_PI);

        SerialBT.print(t);
        SerialBT.print(", ");
        SerialBT.print(hz);
        SerialBT.print("HZ, ");

        SerialBT.print("ypr ");
        SerialBT.print(ypr[0] * 180 / M_PI);
        SerialBT.print(",");
        SerialBT.print(ypr[1] * 180 / M_PI);
        SerialBT.print(",");
        SerialBT.print(ypr[2] * 180 / M_PI);

        //#endif
        //
        //#ifdef OUTPUT_READABLE_REALACCEL
        // display real acceleration, adjusted to remove gravity
        mpu.dmpGetQuaternion(&q, fifoBuffer);
        mpu.dmpGetAccel(&aa, fifoBuffer);
        mpu.dmpGetGravity(&gravity, &q);
        mpu.dmpGetLinearAccel(&aaReal, &aa, &gravity);
        Serial.print("  areal ");
        Serial.print(aaReal.x);
        Serial.print(",");
        Serial.print(aaReal.y);
        Serial.print(",");
        Serial.print(aaReal.z);

        SerialBT.print("  areal ");
        SerialBT.print(aaReal.x);
        SerialBT.print(",");
        SerialBT.print(aaReal.y);
        SerialBT.print(",");
        SerialBT.print(aaReal.z);

        Serial.print(" MPU2: ");
        Serial.print(ax); Serial.print(",");
        Serial.print(ay); Serial.print(",");
        Serial.print(az); Serial.print(",");
        Serial.print(gx); Serial.print(",");
        Serial.print(gy); Serial.print(",");
        Serial.print(gz); Serial.print(",");

        SerialBT.print(" MPU2: ");
        SerialBT.print(ax); SerialBT.print(",");
        SerialBT.print(ay); SerialBT.print(",");
        SerialBT.print(az); SerialBT.print(",");
        SerialBT.print(gx); SerialBT.print(",");
        SerialBT.print(gy); SerialBT.print(",");
        SerialBT.print(gz); SerialBT.print(",");

        Serial.print(" IMU3 ");
        Serial.println(report);

        //***********************************************************************************
        //******************BT***************************************************************
        //***********************************************************************************

        //

        //
        SerialBT.print(" IMU3 ");
        SerialBT.println(report);

        previous = timeNow;

        // blink LED to indicate activity
        blinkState = !blinkState;
        digitalWrite(LED_PIN, blinkState);

        //    Wire.beginTransmission(8); // transmit to device #8
        //
        //    Wire.write("data: ");        // sends five bytes
        //    String outX(ypr[0] * 180 / M_PI);
        //    for (int i = 0; i < outX.length(); i++) {
        //      Wire.write(outX.charAt(i));
        //    }
        //
        //    //Wire.write(out1);
        //    Wire.write(", ");// sends five bytes
        //    String outY(ypr[1] * 180 / M_PI);
        //    for (int i = 0; i < outX.length(); i++) {
        //      Wire.write(outY.charAt(i));
        //    }
        //    Wire.write(",");// sends five bytes
        //    String outZ(ypr[2] * 180 / M_PI);
        //    for (int i = 0; i < outX.length(); i++) {
        //      Wire.write(outZ.charAt(i));
        //    }        // sends five bytes
        //    Wire.endTransmission();    // stop transmitting

      }
    }
  }
}
