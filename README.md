# NetworkSecurityETL

An ETL pipeline for network security data: extract, transform, load, and optionally run ML predictions.

## Features
- Ingest and process network logs  
- Clean, validate, and transform data  
- Store processed data and generate predictions  
- Modular and extensible  

## Setup & Usage

```bash
# Clone the repo
git clone https://github.com/koppolu-buddha-bhavan/NetworkSecurityETL.git
cd NetworkSecurityETL

# (Optional) Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the ETL pipeline
python main.py

# Run specific module (example)
python push_data.py
