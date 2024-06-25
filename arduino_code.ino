// the setup function runs once when you press reset or power the board
int init_flag = 1;
int cali_flag = 0;
int reco_flag = 0;

int cali_count = 0;
int reco_count = 0;
String sensor_val;

void setup() {
    //Initialize the serial port with baud rate of 115200
    Serial.begin(115200);
    Serial1.begin(115200);
}

// the loop function runs over and over again forever
void loop() {
  /*
  while(init_flag == 1){
    Serial.println("10000,10000,10000,10000,10000,");
    delay(5);
  }
  */
  //Serial1.println("No command now");
  
  while(cali_flag == 1){
    Serial1.println("key_pressed for calibration");
    Serial1.println("First collect 100 max value mean");
    while(cali_count < 150)
    {
          for(int i = 0; i < 5; i++){
            sensor_val += (String(10000) + ',');
          }
          sensor_val += "\r\n";
          Serial.print(sensor_val);
          Serial1.print(sensor_val);
          sensor_val = "";
          delay(5);
          cali_count++;
    }
    cali_count = 0;
    while(cali_count < 400)
    {
          for(int i = 0; i < 5; i++){
            sensor_val += (String(8000) + ',');
          }
          sensor_val += "\r\n";
          
          Serial.print(sensor_val);
          Serial1.print(sensor_val);
          sensor_val = "";
          delay(5);
          cali_count++;
    }

    if(cali_count == 400){
      Serial1.println("400 cali data received");
      cali_flag = 0;
      init_flag = 1;
      cali_count = 0;
    }
  }
  for(int i = 0; i < 5; i++){
    sensor_val += (String(10000) + ',');
  }
  sensor_val += "\r\n";
  
  Serial.print(sensor_val);
  sensor_val = "";
  delay(5);
  while(reco_flag == 1){
    // Press the 5 keys in the order of 1-1,2-0,3-0,4-0,5-0
    while(reco_count < 200)
    {
        Serial.println("8000,10000,10000,10000,10000,");
        reco_count++;
        delay(5);
    }
    reco_count = 0;
    while(reco_count < 200)
    {
        Serial.println("10000,8000,10000,10000,10000,");
        reco_count++;
        delay(5);
    }
    reco_count = 0;
    while(reco_count < 200)
    {
        Serial.println("10000,10000,8000,10000,10000,");
        reco_count++;
        delay(5);
    }
    reco_count = 0;
    while(reco_count < 200)
    {
        Serial.println("10000,10000,10000,8000,10000,");
        reco_count++;
        delay(5);
    }
    reco_count = 0;
    while(reco_count < 200)
    {
        Serial.println("10000,10000,10000,10000,8000,");
        reco_count++;
        delay(5);
    }
    reco_count = 0;
    reco_flag = 0;

  }
}
// Add serial interrupt to wait for a 'c' character to start calibration in main function
void serialEvent1(){
  while(Serial1.available()){
    char inChar = (char)Serial1.read();
    Serial1.print("I received: ");
    Serial1.println(inChar);

    //Serial.println(inChar);
    if(inChar == 'c'){
        Serial1.println("calibration command received");
        init_flag = 0;
        cali_flag = 1;
        reco_flag = 0;
    }
    else if(inChar == 's'){
        Serial1.println("stop calibration");
        init_flag = 1;
        cali_flag = 0;
        reco_flag = 0;
    }
    else if(inChar == 'r'){
        Serial1.println("recording command received");
        reco_flag = 1;
        cali_flag = 0;
        init_flag = 0;
    }
    else if(inChar == 'p'){

    }
    else{
      Serial.println("Invalid input");
    }
  }
}
