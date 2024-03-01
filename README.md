# AWS Spark Diabetic Retinopathy Detection

This project utilizes Apache Spark, hosted on an AWS distributed Hadoop cluster, to detect diabetic retinopathy in eye images. By leveraging the scalability of AWS services and the computational power of Spark, the project aims to classify eye images into those affected and unaffected by diabetic retinopathy with high accuracy. This README provides detailed instructions on setting up the environment, running the project, and further developing the classifier.

## Features

- **AWS Hadoop Cluster Integration:** Deployed on AWS to utilize distributed computing for processing large datasets efficiently.
- **Apache Spark for Processing:** Utilizes Spark for data processing and machine learning pipeline execution.
- **Deep Learning for Feature Extraction:** Incorporates a pretrained InceptionV3 model for extracting features from eye images.
- **Logistic Regression for Classification:** Uses logistic regression to classify images, supported by Spark's machine learning library.
- **Performance Evaluation:** Evaluates the model based on accuracy and weighted precision metrics.

## Prerequisites

Ensure you have the following prerequisites installed and configured:

- **AWS Account:** Required to set up the distributed Hadoop cluster using AWS services.
- **Apache Spark:** The core framework used for processing data.
- **Databricks Spark-Deep-Learning Library:** To integrate Spark with deep learning models.
- **Python Libraries:** `numpy`, `pandas`, `opencv-python`, `findspark`, `pyspark`.
- **Hadoop and Spark Setup on AWS:** Cluster configured to run Spark jobs.

## Installation

1. **Configure AWS Hadoop Cluster:**
   - Set up an AWS EMR cluster with Hadoop and Spark. Refer to AWS documentation for detailed instructions.
2. **Spark and Deep Learning Library Setup:**
   - Ensure your Spark session is configured to include the `databricks:spark-deep-learning` package.
   - Example Spark configuration:
     ```
     SparkSession.builder.config('spark.jars.packages', 'databricks:spark-deep-learning:1.5.0-spark2.4-s_2.11').getOrCreate()
     ```
3. **Python Environment:**
   - Install the necessary Python libraries using pip:
     ```
     pip install numpy pandas opencv-python findspark pyspark
     ```

## Usage

1. **Dataset Preparation:**
   - Organize your eye images into two directories: `/data/effected/` for images showing diabetic retinopathy and `/data/uneffected/` for normal images.
2. **Running the Classifier:**
   - Execute the script to read the datasets, process them through the Spark pipeline, and classify the images.
   - The script automatically splits the data into training and testing sets, trains the model, and evaluates its performance.

## Development

- **Improving Model Accuracy:** Experiment with different machine learning models and parameters to improve classification accuracy.
- **Scalability:** Optimize your AWS and Spark configurations for handling larger datasets and achieving faster processing times.
- **Data Augmentation:** Implement image preprocessing and augmentation techniques to enhance model training.

## Contributing

We welcome contributions to enhance the functionality, accuracy, or usability of this project. Please adhere to best practices for code contributions and documentation.

