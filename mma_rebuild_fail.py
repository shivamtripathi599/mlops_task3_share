#!/usr/bin/env python
# coding: utf-8


import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders


#Set up crap for the attachments
files = "/root/mlops_task_3/plots/"
filenames = [os.path.join(files, f) for f in os.listdir(files)]
print(filenames)


#Set up users for email
gmail_user = "emailid@gmail.com"
gmail_pwd = "password"
recipients = ['recipient1@gmail.com','recipient2@gmail.com']

#Create Module
def mail(to, subject, text, attach):
    msg = MIMEMultipart()
    msg['From'] = gmail_user
    msg['To'] = ", ".join(recipients)
    msg['Subject'] = subject
    
    #get all the attachments
    for file in filenames:
        print(file)
        part = MIMEBase('application', 'octet-stream')
        abc = part.set_payload(open(file, 'rb').read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename="%s"'% os.path.basename(file))
        msg.attach(part)

    msg.attach(MIMEText(text))
    mailServer = smtplib.SMTP("smtp.gmail.com", 587)
    mailServer.ehlo()
    mailServer.starttls()
    mailServer.ehlo()
    mailServer.login(gmail_user, gmail_pwd)
    mailServer.sendmail(gmail_user, to, msg.as_string())
    # Should be mailServer.quit(), but that crashes...
    mailServer.close()


#send it
mail(recipients,
   "Rebuild Model Training Report (Fail)",
   "Model training dataset not enough for taining our model for the desired percentage of accuracy. The graphical summary of accuracy and loss with respect to epochs is give below....",
   filenames)



