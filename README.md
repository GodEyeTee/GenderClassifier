# ThaiCarLicensePlate
โปรแกรมนี้จะทำการ detect ใบหน้าบุคคล จากนั้นแสดงเพศของบุคคลนั้นให้มีความแม่นยำมากที่สุด
 
### 6614401149 ญาณกร จารุเดชศิริ
### 6614450085 ธิติมา คำกอง

------------------------------------------------------------------------------------------------------------------------

## โปรแกรมสำหรับส่วนใช้งานจริง
จะอยู่ในโฟลเดอร์ program โดยมีโครงสร้างดังนี้

![alt text]()

โดยแต่ละไฟล์และโฟลเดอร์ จะมีหน้าที่ดังนี้

- โฟลเดอร์ data จะเก็บไฟล์นามสกุล yaml เอาไว้สำหรับใช้กับโมเดล YOLOv5 ในตัวโปรแกรมของเรา
- โฟลเดอร์ models ใช้เก็บโมเดลและโปรแกรมสำหรับการใช้งาน YOLOv5 เอาไว้
- โฟลเดอร์ utils จะเก็บไฟล์ต่างๆที่จำเป็นสำหรับการใช้งาน YOLOv5 เอาไว้
- โฟลเดอร์ weights จะเก็บ weights และ models สำหรับใช้ในตัวโปรแกรมของเราเอาไว้
- ไฟล์ FaceDetector.py ใช้สำหรับ detect ใบหน้าบุคคลบนรูปภาพ โดยจะรับ input เป็นรูปภาพ จากนั้นให้ output เป็นอาเรย์รูปภาพใบหน้าบุคคลที่ detect ได้ในรูปภาพนั้นๆ
- ไฟล์ GenderClassifier.py ใช้สำหรับ clssify เพศ โดยจะรับ input เป็นรูปภาพ จากนั้นให้ output เป็นความเป็นไปได้สำหรับเพศในภาพนั้นๆ คลาสไหนมีค่ามากที่สุดจะตัดสินใจเป็นคลาสนั้น เช่น 0 คือ เพศชาย
- ไฟล์ export.py เป็นไฟล์ที่เอาไว้ใช้เรียกใช้งานโปรแกรม YOLOv5
- ไฟล์ main.py เป็นโปรแกรมหลักในการใช้รันโปรแกรม

โดยจะมีวิธีการทำงานหลักรวมๆ ดังรูปนี้

![alt text]()

------------------------------------------------------------------------------------------------------------------------

## ส่วนของการ Train โมเดล
จะอยู่ในโฟลเดอร์ train โดยมีโครงสร้างดังนี้

![alt text]()
      
โดยแต่ละไฟล์และโฟลเดอร์ จะมีหน้าที่ดังนี้

- โฟลเดอร์ data จะเก็บโฟลเดอร์ดาต้ารูปภาพสำหรับเทรนโมเดล FaceDetector เอาไว้ 
- โฟลเดอร์ lib จะเก็บไฟล์นามสกุล py และ ipynb เอาไว้ โดยในโฟลเดอร์จะประกอบด้วยไฟล์ที่มีลำดับขั้นตอนดังนี้

     1) การเทรน FaceDetector Model
          
          โดย FaceDetector Model จะเป็นโมเดลที่ใช้ detect หาใบหน้า 
     
          เนื่องจากเป็นการใช้โมเดล YOLOv5 มาทำ Transfer Learning จึงแนะนำให้เทรนไฟล์ *TrainYolov5Face.ipynb*  บน colab 
          เป็นไฟล์สำเร็จรูปที่พร้อมรันเทรนได้เลย มีการนำดาต้ารูปภาพมาจาก Roboflow ซึ่งผู้จัดทำได้ทำการ annotation พร้อมเทรนเรียบร้อย 
          หลังจากเทรนเสร็จแล้ว weight ที่พร้อมใช้งานจะถูกดาวน์โหลดลงคอมพิวเตอร์ของผู้เทรนในรูปแบบของไฟล์นามสกุล pt

     2) การเทรน GenderClassifier Model
        
          โดย GenderClassifier Model จะเป็นโมเดลที่ใช้ classify เพศ

          2.1) *split_crop_AgeAndGender.py* จะทำหน้าที่จัดการดาต้าจากโฟลเดอร์ InthewildFaces มารวบรวมให้เป็นโฟลเดอร์ New_InTheWildFaces พร้อมทั้งจัดการเพิ่มครอบให้เหลือเพียงภาพใบหน้าและเลือกช่วงอายุเป็น 20-60 ปี 

          2.2) *splitTrainTestVal.py* จะทำหน้าที่นำดาต้ารูปภาพจากโฟลเดอร์ New_InTheWildFaces มารวบรวมเป็นโฟลเดอร์ Split_New_InTheWildFaces ที่มีการแบ่งดาต้าเป็น Train , Test และ Validation ให้พร้อมสำหรับการเข้าเทรนโมเดล

          2.3) *TrainGenderClassifierModel.ipynb* จะเป็นไฟล์ที่ทำการเทรนโมเดล GenderClassifier ด้วยการใช้ไลบรารี Tensorflow และ Keras เมื่อรันครบทุก shell ผลลัพธ์จะออกมาเป็นไฟล์นามสกุล h5 ในโฟลเดอร์ models

          และเนื่องจาก Github นั้นไม่สามารถอัพโหลดไฟล์หรือโฟลเดอร์ที่มีขนาดใหญ่มากได้ เราจึงอัพโหลดภาพดาต้าตัวอย่างลงในโฟลเดอร์ data 

- โฟลเดอร์ models จะเก็บโมเดลที่ชื่อว่า model_150x200.h5 ที่ได้จากการเทรน GenderClassifier Model เอาไว้

------------------------------------------------------------------------------------------------------------------------

## ความต้องการ
* Python 3.7 or later

------------------------------------------------------------------------------------------------------------------------

## วิธีการใช้งาน

โปรแกรมนี้จะมีการทำงานแบบ real-time ผ่านกล้องคอมพิวเตอร์ของผู้ใช้

1.) clone ตัวโปรเจคนี้ด้วยคำสั่ง

     git clone https://github.com/GodEyeTee/GenderClassifier.git
     
2.) เข้าไปที่โฟลเดอร์ GenderClassifier

     cd GenderClassifier

3.) ติดตั้ง library ที่จำเป็น ด้วยการรันคำสั่ง 
     
     pip install -r requirements.txt
     
4.) เข้าไปที่โฟลเดอร์ program ของตัวโปรเจค

     cd program

5.) รันโปรแกรมหลัก ด้วยคำสั่ง

     python main.py

6.) เมื่อรันแล้ว โปรแกรมจะโชว์ผลลัพธ์ให้ดูแบบ real-time จากกล้องของผู้ใช้





