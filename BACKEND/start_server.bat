start cmd /k python app.py
timeout /t 5
start cmd /k ngrok http --domain=sound-caring-rhino.ngrok-free.app 80
