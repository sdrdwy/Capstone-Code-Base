- Env Setup: 
    ```bash
    conda create -n your_name python=3.10
    conda activate your_name
    pip install -U -r requirements.txt
    ```
    - Create file `.env`, content:
    ```bash
    DASHSCOPE_API_KEY=sk-xxx
    NEO4J_URI=your_db_url
    NEO4J_USERNAME=your_db_usrname
    NEO4J_PASSWORD=your_passwd
    DB_NAME=your_db_name
    ```
