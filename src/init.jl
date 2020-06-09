function __init__()
    register(DataDep("VarianceComponentsHDFE",
        "Downloading medium_main.csv test data",
        ["https://vchdfe.s3-us-west-2.amazonaws.com/medium_main.csv.tar.gz",
        "https://vchdfe.s3-us-west-2.amazonaws.com/medium_controls_main.csv.tar.gz"],
        ["3dea578dd21c78e8cb148d19067638222a6c4be886a805272ae9d4a576f81042",
        "ad308402a5a63e58eef1dd85316a42e96789391acde9d97897f33c08289681df"];
        post_fetch_method= [unpack, unpack]
    ))
    return nothing
end
# "3dea578dd21c78e8cb148d19067638222a6c4be886a805272ae9d4a576f81042";
