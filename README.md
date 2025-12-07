# CarPricePred

# Setup:
## Backend:
Create a new environment:
```
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```
## Frontend:
Install nvm:<br>
For MacOS:<br>
Please refer to https://github.com/nvm-sh/nvm<br>
For Windows:<br>
Download and install https://github.com/coreybutler/nvm-windows<br>
Then run:
```
nvm install 22
uvm use 22
cd frontend
npm install
```
# How to Run:
## Method 1: One-Click Launch (Recommended)
We provide a convenient script to launch both frontend and backend simultaneously.

Windows users: <br>
Double-click ***run_app.bat*** in the project root directory. <br>
(This will open two command prompt windows displaying backend and frontend logs respectively.)

Linux / macOS users: <br>
Run in Terminal:
```
chmod +x run_app.sh
./run_app.sh
```
## Method 2: Manual Startup
### If you need to debug separately, run the following in **two terminal** windows: <br>
### Terminal 1 (Backend):
#### Ensure the virtual environment is activated!!!
#### Ensure you are in the **project root** directory!!!
```
python -m backend.main
```
The backend runs by default at: http://localhost:8000
### Terminal 2 (Frontend):
```
cd frontend
npm run dev
```
The frontend runs by default at: http://localhost:5173/

## Data Source:
    used_car.csv : https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset
    Car details v3.csv  
    car details v4.csv  
        Source: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
