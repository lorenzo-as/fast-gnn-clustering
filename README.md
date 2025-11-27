# fast-gnn-clustering

- Install requirements
    ```
    pip install -r ./requirements.txt
    ```

- Clone and install the ```quantized-gravnet``` repository
    ```
    git clone https://github.com/lorenzo-as/quantized-gravnet.git
    cd quantized-gravnet
    git checkout dev
    pip install -e .
    cd ..
    ```

- Fetch from Zenodo and preprocess data + train a model
    ```
    python preprocess.py
    python train.py
    ```
