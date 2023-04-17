# VA53-Kalman-Filter

## Description

University practical work to implement a basic people tracking using predictive Kalman Filter. 

The aim was to try different models (Position, Speed, Acceleration) at different video input framerate.

As the body detection was not the main goal of this work, <u>it has been done roughly and is not fine-tuned for every video input.</u>

The report written for evaluation (in french) as well as the measurements sheet are available under `/docs`. 

The images generated for illustration purpose are under `/img`, where **Green line** is the <u>prediction</u> and **Red line** the <u>detection</u>.

> Note : the input data used in this project has been removed for pirvacy purpose.

## How to use

Just run `kalman.py` with Python 3.

You can input file and switch the Kalman model by editing inside the source code.
