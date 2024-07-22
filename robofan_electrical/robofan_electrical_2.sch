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
L Connector:Raspberry_Pi_2_3 J?
U 1 1 603C5B02
P 3200 3500
F 0 "J?" H 3200 4981 50  0000 C CNN
F 1 "Raspberry_Pi_2_3" H 3200 4890 50  0000 C CNN
F 2 "" H 3200 3500 50  0001 C CNN
F 3 "https://www.raspberrypi.org/documentation/hardware/raspberrypi/schematics/rpi_SCH_3bplus_1p0_reduced.pdf" H 3200 3500 50  0001 C CNN
	1    3200 3500
	1    0    0    -1  
$EndComp
$Comp
L misc:BIG_EASY_DRIVER U?
U 1 1 603C7B3E
P 6900 3650
F 0 "U?" H 6900 4347 60  0000 C CNN
F 1 "BIG_EASY_DRIVER" H 6900 4241 60  0000 C CNN
F 2 "" H 6900 3650 50  0001 C CNN
F 3 "" H 6900 3650 50  0001 C CNN
	1    6900 3650
	1    0    0    -1  
$EndComp
Wire Wire Line
	6200 4100 4700 4100
Wire Wire Line
	4700 4100 4700 3200
Wire Wire Line
	4700 3200 4000 3200
Wire Wire Line
	6200 4000 5300 4000
Wire Wire Line
	5300 4000 5300 5050
Wire Wire Line
	5300 5050 2050 5050
Wire Wire Line
	2050 5050 2050 3000
Wire Wire Line
	2050 3000 2400 3000
Wire Wire Line
	6200 3700 5400 3700
Wire Wire Line
	5400 3700 5400 5150
Wire Wire Line
	5400 5150 1950 5150
Wire Wire Line
	1950 5150 1950 3700
Wire Wire Line
	1950 3700 2400 3700
Wire Wire Line
	6200 3900 5500 3900
Wire Wire Line
	5500 3900 5500 5250
Wire Wire Line
	5500 5250 3500 5250
Wire Wire Line
	3500 5250 3500 4900
Wire Wire Line
	2800 4900 2800 4800
Wire Wire Line
	2900 4800 2900 4900
Connection ~ 2900 4900
Wire Wire Line
	2900 4900 2800 4900
Wire Wire Line
	3000 4800 3000 4900
Connection ~ 3000 4900
Wire Wire Line
	3000 4900 2900 4900
Wire Wire Line
	3100 4800 3100 4900
Connection ~ 3100 4900
Wire Wire Line
	3100 4900 3000 4900
Wire Wire Line
	3200 4800 3200 4900
Wire Wire Line
	3100 4900 3200 4900
Connection ~ 3200 4900
Wire Wire Line
	3200 4900 3300 4900
Wire Wire Line
	3300 4800 3300 4900
Connection ~ 3300 4900
Wire Wire Line
	3300 4900 3400 4900
Wire Wire Line
	3400 4800 3400 4900
Connection ~ 3400 4900
Wire Wire Line
	3400 4900 3500 4900
Connection ~ 3500 4900
Wire Wire Line
	3500 4800 3500 4900
Wire Wire Line
	2400 3800 1850 3800
Wire Wire Line
	1850 3800 1850 5350
Wire Wire Line
	1850 5350 5650 5350
Wire Wire Line
	5650 5350 5650 3600
Wire Wire Line
	5650 3600 6200 3600
Wire Wire Line
	2400 3900 1750 3900
Wire Wire Line
	1750 3900 1750 5450
Wire Wire Line
	1750 5450 5750 5450
Wire Wire Line
	5750 5450 5750 3500
Wire Wire Line
	5750 3500 6200 3500
Wire Wire Line
	2400 4000 1650 4000
Wire Wire Line
	1650 4000 1650 5550
Wire Wire Line
	1650 5550 5850 5550
Wire Wire Line
	5850 5550 5850 3400
Wire Wire Line
	5850 3400 6200 3400
Wire Wire Line
	4000 3300 6200 3300
Wire Wire Line
	4000 3400 5300 3400
Wire Wire Line
	5300 3400 5300 3200
Wire Wire Line
	5300 3200 6200 3200
$EndSCHEMATC
