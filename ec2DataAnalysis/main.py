import datetime
import numpy as np
import scipy.stats as st
import pandas as pd
import ydata_profiling as pp
import matplotlib.pyplot as plt


# Print messages with date and time information
def print_message(message):
    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(now + " - " + message)


# The Main Procedure
if __name__ == '__main__':

    # Instance type and region for analysis
    instance_type = "c4.2xlarge"
    region_filter = "us-east-1c"

    print_message("Beginning the data analysis procedure")

    print_message("Step 1: reading data from multiple files")
    column_names = ["Label","RegionDC","Instance","OpSystem","Price","Timestamp"]
    df1 = pd.read_csv("./data/202001/" + instance_type + ".txt", sep='\t', header=None, names=column_names)
    df2 = pd.read_csv("./data/202002/" + instance_type + ".txt", sep='\t', header=None, names=column_names)
    df3 = pd.read_csv("./data/202003/" + instance_type + ".txt", sep='\t', header=None, names=column_names)
    df4 = pd.read_csv("./data/202004/" + instance_type + ".txt", sep='\t', header=None, names=column_names)
    df5 = pd.read_csv("./data/202005/" + instance_type + ".txt", sep='\t', header=None, names=column_names)
    df6 = pd.read_csv("./data/202006/" + instance_type + ".txt", sep='\t', header=None, names=column_names)
    df7 = pd.read_csv("./data/202007/" + instance_type + ".txt", sep='\t', header=None, names=column_names)
    df8 = pd.read_csv("./data/202008/" + instance_type + ".txt", sep='\t', header=None, names=column_names)
    df9 = pd.read_csv("./data/202009/" + instance_type + ".txt", sep='\t', header=None, names=column_names)
    df10 = pd.read_csv("./data/202010/" + instance_type + ".txt", sep='\t', header=None, names=column_names)
    df11 = pd.read_csv("./data/202011/" + instance_type + ".txt", sep='\t', header=None, names=column_names)
    df12 = pd.read_csv("./data/202012/" + instance_type + ".txt", sep='\t', header=None, names=column_names)

    print_message("Step 2: concatenating data")
    df = pd.concat([df1, df2, df3, df4, df5, df6, df8, df9, df10, df11, df12])

    print_message("Step 3: adjusting data")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%dT%H:%M:%S.000Z")

    print_message("Step 4: visualizing head of data")
    print(df.info())
    print(df.head())

    print_message("Step 5: generating data profile report")
    prof = pp.ProfileReport(df, correlations={"auto": {"calculate": False}})
    prof.to_file('./report/pandas_profile_report.html')

    print_message("Step 5: filtering data")
    condition = df["RegionDC"] == region_filter
    filtered_df = df.where(condition)

    print_message("Step 6: generating visualization report on filtered data")
    filtered_df.plot.scatter(x="Timestamp", y="Price", figsize=(11, 8))
    plt.title("Spot Price Variations for " + instance_type + " on region " + region_filter, size=12)
    plt.xlabel("Timestamp", size=10)
    plt.ylabel("Price", size=10)
    plt.grid(visible=True)
    plt.savefig("./report/histogram_filtered.pdf", dpi=300)

    print_message("Step 7: generating visualization report")
    fig, ax = plt.subplots()
    for key, grp in df.groupby(["RegionDC"]):
        ax = grp.plot(ax=ax, kind="line", x="Timestamp", y="Price", label=key, figsize=(11, 8))
    plt.title("Spot Price Variations for " + instance_type, size=12)
    plt.xlabel("Timestamp", size=10)
    plt.ylabel("Price", size=10)
    plt.legend(loc="best")
    plt.grid(visible=True)
    plt.savefig("./report/histogram.pdf", dpi=300)

    print_message("Step 8: Confidence interval for price mean")
    ci90 = st.norm.interval(alpha=0.90, loc=np.mean(df["Price"]), scale=st.sem(df["Price"]))
    ci95 = st.norm.interval(alpha=0.95, loc=np.mean(df["Price"]), scale=st.sem(df["Price"]))
    print("CI90% = ", end="")
    print(ci90)
    print("CI95% = ", end="")
    print(ci95)

    print_message("Finishing the data analysis procedure")
