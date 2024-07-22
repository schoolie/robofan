EESchema Schematic File Version 4
EELAYER 30 0
EELAYER END
$Descr A4 11693 8268
encoding utf-8
Sheet 1 1
Title ""
Date ""
Rev ""
Comp ""
Comment1 ""
Comment2 ""
Comment3 ""
Comment4 ""
$EndDescr
$Comp
L misc:BIG_EASY_DRIVER U?
U 1 1 6037143C
P 6000 3850
F 0 "U?" H 6000 4547 60  0000 C CNN
F 1 "BIG_EASY_DRIVER" H 6000 4441 60  0000 C CNN
F 2 "" H 6000 3850 50  0001 C CNN
F 3 "" H 6000 3850 50  0001 C CNN
	1    6000 3850
	1    0    0    -1  
$EndComp
Wire Wire Line
	7650 3500 7650 3450
Wire Wire Line
	6700 3400 7300 3400
Wire Wire Line
	6700 3600 7300 3600
$Comp
L Motor:Stepper_Motor_bipolar M?
U 1 1 60370C55
P 7600 3500
F 0 "M?" H 7788 3624 50  0000 L CNN
F 1 "Stepper_Motor_bipolar" H 7788 3533 50  0000 L CNN
F 2 "" H 7610 3490 50  0001 C CNN
F 3 "http://www.infineon.com/dgdl/Application-Note-TLE8110EE_driving_UniPolarStepperMotor_V1.1.pdf?fileId=db3a30431be39b97011be5d0aa0a00b0" H 7610 3490 50  0001 C CNN
	1    7600 3500
	1    0    0    -1  
$EndComp
Wire Wire Line
	6700 3700 8150 3700
Wire Wire Line
	8150 3700 8150 3200
Wire Wire Line
	8150 3200 7700 3200
Wire Wire Line
	6700 3500 6850 3500
Wire Wire Line
	6850 3500 6850 3200
Wire Wire Line
	6850 3200 7500 3200
$Comp
L power:+12V #PWR?
U 1 1 6037507B
P 7050 3800
F 0 "#PWR?" H 7050 3650 50  0001 C CNN
F 1 "+12V" V 7065 3928 50  0000 L CNN
F 2 "" H 7050 3800 50  0001 C CNN
F 3 "" H 7050 3800 50  0001 C CNN
	1    7050 3800
	0    1    1    0   
$EndComp
Text Label 7100 3400 0    50   ~ 0
R
Text Label 7100 3200 0    50   ~ 0
B
Text Label 7100 3600 0    50   ~ 0
Bk
Text Label 7100 3700 0    50   ~ 0
G
$Comp
L Connector:Raspberry_Pi_2_3 J?
U 1 1 6037794C
P 3550 3700
F 0 "J?" H 3550 5181 50  0000 C CNN
F 1 "Raspberry_Pi_2_3" H 3550 5090 50  0000 C CNN
F 2 "" H 3550 3700 50  0001 C CNN
F 3 "https://www.raspberrypi.org/documentation/hardware/raspberrypi/schematics/rpi_SCH_3bplus_1p0_reduced.pdf" H 3550 3700 50  0001 C CNN
	1    3550 3700
	1    0    0    -1  
$EndComp
Text Label 2650 3300 0    50   ~ 0
R
Wire Wire Line
	3350 5000 3350 5250
Text Label 3350 5200 0    50   ~ 0
Br
Text Label 2650 4000 0    50   ~ 0
O
Text Label 2650 4100 0    50   ~ 0
Y
Wire Wire Line
	2050 4100 2050 2100
Wire Wire Line
	2050 2100 5300 2100
Wire Wire Line
	5300 2100 5300 3400
Wire Wire Line
	2050 4100 2750 4100
Wire Wire Line
	3350 5250 4600 5250
$Comp
L power:GND #PWR?
U 1 1 603752C6
P 7050 4000
F 0 "#PWR?" H 7050 3750 50  0001 C CNN
F 1 "GND" H 7055 3827 50  0000 C CNN
F 2 "" H 7050 4000 50  0001 C CNN
F 3 "" H 7050 4000 50  0001 C CNN
	1    7050 4000
	1    0    0    -1  
$EndComp
Wire Wire Line
	7050 3800 6700 3800
Wire Wire Line
	6700 3950 7050 3950
Wire Wire Line
	7050 3950 7050 4000
Wire Wire Line
	5300 4100 4600 4100
Wire Wire Line
	4600 4100 4600 5250
Wire Wire Line
	2300 3300 2300 5350
Wire Wire Line
	2300 5350 4700 5350
Wire Wire Line
	4700 5350 4700 4200
Wire Wire Line
	4700 4200 5300 4200
Wire Wire Line
	2300 3300 2750 3300
Wire Wire Line
	5300 4300 4800 4300
Wire Wire Line
	4800 4300 4800 5450
Wire Wire Line
	4800 5450 2400 5450
Wire Wire Line
	2400 5450 2400 4000
Wire Wire Line
	2400 4000 2750 4000
Text Label 5150 4300 0    50   ~ 0
O
Text Label 5150 4200 0    50   ~ 0
R
Text Label 5300 3250 0    50   ~ 0
Y
Wire Wire Line
	3650 2400 3650 2200
Wire Wire Line
	3650 2200 5000 2200
Wire Wire Line
	5000 2200 5000 4000
Wire Wire Line
	5000 4000 5300 4000
$EndSCHEMATC
