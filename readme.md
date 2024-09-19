# Worker safety detection app which runs on AWS Panorama edge device

This is a worker safety detection app that runs on AWS Panorama edge device. The app detects if a worker is wearing a helmet and a safety vest. The app uses a pre-trained model to detect the helmet and safety vest. The app uses the AWS Panorama device SDK to interact with the device.

## Prerequisites
1. AWS Panorama device
2. AWS Panorama device SDK
3. AWS Panorama device certificate
4. AWS Panorama device private key
5. AWS Panorama device root CA certificate
6. AWS Panorama device model package
7. AWS Panorama device model package manifest

## Installation
1. Build the app on ARM architecture EC2 instance
2. Package the app
3. Deploy the app on AWS Panorama device using AWS console