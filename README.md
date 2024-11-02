1. Create a virtual environment
```sh
python -m venv env
```

2. Activate the environment
```sh
# Windows
.\env\Scripts\activate
# Linux
source env/bin/activate
```

3. Install the project dependencies
```sh
pip install -r requirements.txt
```

4. Run the server
```sh
uvicorn main:app --reload
```

http://localhost:8000