#!/usr/local/bin/python

#A
import RPi.GPIO as GPIO
import time
import urllib.request
import requests
import time
from difflib import SequenceMatcher
from gpiozero import Servo

#N
import os
import ocr

#R
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer 
import pandas as pd
from pathlib import Path
from twilio.rest import Client
from oauth2client.service_account import ServiceAccountCredentials
import pygsheets

# Google Sheets stuff:

scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name(
    'client_secrets.json', scope)

spreadsheet_key = '190g032Wi1UO8RdGUmunuQvR6CDv-5VuuwQ-7bbCwDvM'
# Sheet name not the workbook
wks_name = 'Spam/Ham'

# Credentials in the form of a 'client_secret.json' file can be retrieved from GCP using the Google Sheets API
gc = pygsheets.authorize(service_file='client_secrets.json')

sh = gc.open_by_key(spreadsheet_key)

# Twilio (sms) stuff:

# Your Account SID from twilio.com/console
account_sid = ""
# Your Auth Token from twilio.com/console
auth_token  = ""

client = Client(account_sid, auth_token)

# Dataset preprocessing for ML:

#Read data
df = pd.read_csv('sms.csv',usecols = [0,1],encoding='latin-1' )
#Rename columns
df.rename(columns = {'v1':'Category','v2': 'Message'}, inplace = True)
#Create the standard column from category column
df['type']=df.apply(lambda row: 1 if row.Category=='ham' else 0, axis=1)
#print(df)

X=list(df['Message'])
#print(X)
y=list(df['type'])
#print(y)


knn = KNeighborsClassifier(n_neighbors=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
vect = CountVectorizer()
vect.fit(X_train)
CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)
knn.fit(X_train_dtm,y_train)
KNeighborsClassifier(n_neighbors=1)
y_pred = knn.predict(X_test_dtm)

#A
GPIO.setmode(GPIO.BOARD)
servoPIN = 27
yellowPIN = 22
redPIN = 23
greenPIN = 24
z = 1

#define the pin that goes to the circuit
pin_to_circuit = 7
def rc_time (pin_to_circuit):
    count = 0
	
	#Output on the pin for 
    GPIO.setup(pin_to_circuit, GPIO.OUT)
    GPIO.output(pin_to_circuit, GPIO.LOW)
    time.sleep(0.1)
    
    #Change the pin back to input
    GPIO.setup(pin_to_circuit, GPIO.IN)
    
    #Count until the pin goes high
    while (GPIO.input(pin_to_circuit) == GPIO.LOW):
        count += 1
        
    return count
	
#Catch when script is interrupted, cleanup correctly
old_state = 0 #Bright
new_state = 0
count2 = 0
num=1
Extension = ".jpg"
Name = str(num) + Extension

urlcap = 'http://160.39.243.143/capture'
urlsave = 'http://160.39.243.143/saved-photo'

try:
    # Main loop
    while True:
        count = rc_time(pin_to_circuit)
        print(count)
        if count > 300000:
            new_state = 1
            if new_state != old_state:
                print('Dim Light => New Letter?')
                old_state = new_state
            count2 += 1
            if count2 >= 5:
                print('New Letter')
                #Take photograph, Send to Server, Servo Motor, LEDs
                res = requests.get(urlcap)
                print(res.text)
                time.sleep(3)
                urllib.request.urlretrieve(urlsave, Name)
                
                #Nihar's code
                #os.system("sudo python ./ocr.py -i 1.jpg")
                ocr.ocr_code()
                
                GPIO.cleanup()
                GPIO.setmode(GPIO.BCM)
                if z==1:
                    servo = Servo(27)
                    z+=1
                GPIO.setup(servoPIN, GPIO.OUT)
                GPIO.setup(yellowPIN, GPIO.OUT)
                GPIO.setup(redPIN, GPIO.OUT)
                GPIO.setup(greenPIN, GPIO.OUT)
                GPIO.output(yellowPIN, GPIO.LOW)
                
                file = open("/media/usb0/user.txt", "r")
                user_name = file.readline()
                user_name = user_name.lower()
                #user_pin = int(file.readline())
                file.close()

                #print (user_name, user_pin)

                file = open("/home/pi/output/receiver.txt", "r")
                recv_name = file.readline()
                recv_name = recv_name.lower()
                #recv_pin = int(file.readline())
                file.close()

                #print (recv_name, recv_pin)

                x = SequenceMatcher(a=user_name,b=recv_name).ratio()
                print(x)

                #Checking for Current Resident
                name2 = "Current Resident"
                y = SequenceMatcher(a=name2,b=recv_name).ratio()
                print(y)
                
                #if user_name==recv_name:
                if x>0.7 or y>0.7:
                    GPIO.output(greenPIN,GPIO.HIGH)
                    GPIO.output(yellowPIN,GPIO.LOW)
                    print("Mail Receiver is correct!")
                    servo.min()
                    time.sleep(2)
                    #servo.min()
                    servo.max()
                    time.sleep(2)
                    GPIO.output(greenPIN,GPIO.LOW)
                    GPIO.output(yellowPIN,GPIO.HIGH)
                    #GPIO.cleanup()
                    
                    #Rohan's code
                    
                    with open("/home/pi/output/OCR_text.txt") as myfile:
                        head = [next(myfile) for x in range(7)]
                        # Preprocess the data.
                        
                        timestamp = head[0]
                        send_name = head[1]
                        send_add1 = head[2]
                        send_add2 = head[3]
                        rec_name = head[4]
                        rec_add1 = head[5]
                        rec_add2 = head[6]
                        #print(rec_name)
                        step_0 = timestamp.split(' ')
                        dates = [step_0[0]]
                        times = [step_0[1].strip('\n')]
                        #print(dates)
                        step_1 = rec_name.split(': ')
                        receiver_name = step_1[1]
                        #print(receiver_name)
                        step_2 = send_name.split(': ')
                        sender_name = step_2[1]
                        #print(sender_name)
                        step_3 = send_add1.split(': ')
                        sender_address1 = step_3[1]
                        sender_address2 = send_add2
                        #print(sender_address1)
                        #print(sender_address2)
                        step_4 = rec_add1.split(': ')
                        receiver_address1 = step_4[1]
                        receiver_address2 = rec_add2
                        #print(receiver_address1)
                        #print(receiver_address2)
                        test0=[timestamp.strip('\n')]
                        test1=[sender_name.strip('\n')]
                        test2=[receiver_name.strip('\n')]
                        test3=sender_address1.strip('\n')
                        test4=sender_address2.strip('\n')
                        sender_address = [test3 + test4]
                        test5=receiver_address1.strip('\n')
                        test6=receiver_address2.strip('\n')
                        receiver_address = [test5 + test6]
                        #print(receiver_address)

                        # Create a dataframe object that holds all the mail info.

                        d = {
                            "Date": dates,
                            "Time": times,
                            "Sender Name": test1,
                            "Sender Address": sender_address,
                            "Receiver Name": test2,
                            "Receiver Address": receiver_address
                        }
                        dtm2 = vect.transform(test2)
                        dtm1 = vect.transform(test1)

                        # Perform the ML and send a text notifying the user of new mail.

                        pred1 = knn.predict(dtm1.toarray())
                        pred2 = knn.predict(dtm2.toarray())

                        if (pred1==0) and (pred2==0):
                            message = client.messages.create(
                                to="+13473014434",
                                from_="+18647400902",
                                body = "You have new mail from: "+test1[0]+ " " + " - This message is spam." + " " +
                                "Check your stats here: https://datastudio.google.com/reporting/5baac953-c958-40e0-b344-6f94c4c86df4")
                            d['Spam/Ham'] = "Spam"
                        elif (pred1==1) and (pred2==0):
                            message = client.messages.create(
                                to="+13473014434",
                                from_="+18647400902",
                                body = "You have new mail from: "+test1[0]+ " " + 
                                " - This message is spam." + " " +
                                "Check your stats here: https://datastudio.google.com/reporting/5baac953-c958-40e0-b344-6f94c4c86df4")
                            d['Spam/Ham'] = "Spam"
                        elif (pred1==0) and (pred2==1):
                            message = client.messages.create(
                                to="+13473014434",
                                from_="+18647400902",
                                body = "You have new mail from: "+test1[0]+ " " + 
                                " - This message is spam." + " " +
                                "Check your stats here: https://datastudio.google.com/reporting/5baac953-c958-40e0-b344-6f94c4c86df4")
                            d['Spam/Ham'] = "Spam"
                        else:
                            message = client.messages.create(
                                to="+13473014434",
                                from_="+18647400902",
                                body = "You have new mail from: "+test1[0]+ " " + " - This message is not spam." + " " +
                                "Check your stats here: https://datastudio.google.com/reporting/5baac953-c958-40e0-b344-6f94c4c86df4")
                            d['Spam/Ham'] = "Not Spam"
                        print(message.sid)

                        # Update the dataframe object with new records.

                        new_data = pd.DataFrame(d)

                        if Path('test.pkl').is_file():
                            curr_data = pd.read_pickle('test.pkl')
                            updated = pd.concat([curr_data, new_data])
                            updated = pd.DataFrame(updated)
                        else:
                            updated = new_data.copy()

                        updated.to_pickle('test.pkl')

                        # Send the data to Google Sheets to perform data viz. The report can be emailed on schedule.

                        try:
                            sh.add_worksheet(wks_name)
                        except:
                            pass
                        wks_write = sh.worksheet_by_title(wks_name)
                        wks_write.clear('A1',None,'*')
                        wks_write.set_dataframe(updated, (1,1), encoding='utf-8', fit=True)
                        wks_write.frozen_rows = 1
                    
                else:
                    GPIO.output(redPIN,GPIO.HIGH)
                    GPIO.output(yellowPIN,GPIO.LOW)
                    print("Mail Receiver may not be correct")
                    servo.max()
                    time.sleep(2)
                    servo.max()
                    time.sleep(2)
                    GPIO.output(redPIN,GPIO.LOW)
                    GPIO.output(yellowPIN,GPIO.HIGH)
                    #GPIO.cleanup()
                GPIO.cleanup()
                GPIO.setmode(GPIO.BOARD)
                time.sleep(5)
                
        else:
            new_state = 0
            if new_state != old_state:
                print('Bright Light => No New Letter?')
                old_state = new_state
            count2=0
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()
