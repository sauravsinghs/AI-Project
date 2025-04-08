# Hamming Weight Tree Visualization App

This Streamlit application provides an interactive interface for working with Hamming Weight Trees. It allows you to create, visualize, and perform operations on Hamming Weight Trees in real-time.

## Features

- Create and configure Hamming Weight Trees with customizable parameters
- Insert binary codes into the tree
- Perform r-neighbor and k-NN searches
- Visualize the tree structure using network graphs
- View performance statistics and metrics

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

To run the Streamlit app, execute the following command in your terminal:

```bash
streamlit run app.py
```

## Usage

1. **Configure Tree Parameters**

   - Use the sidebar to set the code length, node threshold, number of substrings, and forest size
   - These parameters determine the structure and behavior of your Hamming Weight Tree

2. **Insert Binary Codes**

   - Enter binary codes in the text area (one code per line)
   - Click "Insert Codes" to add them to the tree
   - The app will validate the codes and show any errors

3. **Search Operations**

   - Perform r-neighbor searches by entering a query code and radius
   - Perform k-NN searches by entering a query code and k value
   - Results will be displayed in a table format

4. **Visualization**

   - The tree structure is automatically visualized using a network graph
   - Nodes represent tree nodes, and edges show the relationships between them
   - The visualization updates as you modify the tree

5. **Performance Statistics**
   - View real-time statistics about tree operations
   - Track insertions, searches, cache hits, and cache misses
   - Monitor the performance metrics through the bar chart

## Notes

- All binary codes must have the same length as specified in the tree parameters
- The app provides immediate feedback for invalid inputs
- Performance statistics are reset when you create a new tree
