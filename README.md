# TBHRP -  Hierarchical Risk Parity with Text-Based Clustering

TBHRP (Text-Based Hierarchical Risk Parity) is a portfolio optimization method that extends the traditional Hierarchical Risk Parity (HRP) algorithm. It enhances HRP's clustering step by grouping assets based on the textual similarity of their financial filings (specifically, Item 1 Business Descriptions from 10-K reports, but nearly all text could be used) using TF-IDF.

HRP is a robust alternative to mean-variance optimization because it does not require the explicit inversion of the covariance matrix, making it less sensitive to estimation errors. TBHRP aims to further improve diversification by clustering companies based on fundamental business similarity rather than just price correlation.

## Getting Started

### Prerequisites
- Python 3.x
- `pandas`
- `requests`
- `numpy`
- `scikit-learn`
- `scipy`
- `sec_api` (Requires an API key from sec-api.io)
- `matplotlib` (Optional, for plotting dendrograms)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/bkrayfield/TBHRP.git
   cd TBHRP
   ```
2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
### API Key

You will need an API key from sec-api.io. Once you have your API key, update the API_KEY variable in doHRP.py and classhrp.py.

## Example Scripts!

You can find an example notebook in the examples folder.
* Running TBHRP Example

## Data Requirements

The doHRP.py script requires a CSV file containing historical stock prices. By default, it looks for a file named top500Total.csv.

CSV File Format (top500Total.csv):
idx (Date)	Ticker1	Ticker2	Ticker3	...
2020-01-01	100.00	50.00	200.00	...
2020-01-02	101.50	51.25	198.00	...
...	...	...	...	...

The first column must be a date index.

Subsequent columns must be ticker symbols with their corresponding closing prices.

If your data file has a different name or structure, you must modify the df = pd.read_csv(...) line in doHRP.py to point to your data source.

### Running the Scripts

1. Generate Similarity Matrices and Weights:
    Execute doHRP.py to fetch filing data, compute similarity matrices, and calculate portfolio weights for different strategies. This script will save the results as pickle files (<rnum>pandas_frameswreturns.pk and <rnum>weights.pk, where rnum is a random number generated in the script). You can run the script simply with the following code.
    ```bash
    from TBHRP import doHRP

# Execute the analysis
doHRP.run_hrp_analysis(
    api_key=SEC_API_KEY,
    input_csv_path=INPUT_DATA,
    output_dir=OUTPUT_LOCATION,
    run_id=RUN_IDENTIFIER,
    years=ANALYSIS_YEARS,
    sample_size=10,
    window_size=60,
)
    ```

2. Analyze Results:
    Execute results_.py to load the saved results and calculate performance metrics. Make sure to adjust the home_dir_frames and home_dir_weights variables in results_.py to the directory where your pickle files are saved.
    ```
    python results_.py
    ```

Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for any bugs, feature requests, or improvements.
License

This project is open-source and available under the MIT License.