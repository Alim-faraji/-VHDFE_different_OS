function __init__()
    register(DataDep("VarianceComponentsHDFE",
        "Downloading medium_main.csv, medium_controls_main.csv, full_main.csv, and full_controls_main.csv test data",
        ["https://vchdfe.s3-us-west-2.amazonaws.com/medium_main.csv.tar.gz",
        "https://vchdfe.s3-us-west-2.amazonaws.com/medium_controls_main.csv.tar.gz",
        "https://vchdfe.s3-us-west-2.amazonaws.com/full_main.csv.tar.gz",
        "https://vchdfe.s3-us-west-2.amazonaws.com/full_controls_main.csv.tar.gz"],
        ["3dea578dd21c78e8cb148d19067638222a6c4be886a805272ae9d4a576f81042",
        "ad308402a5a63e58eef1dd85316a42e96789391acde9d97897f33c08289681df",
        "a0d64c1d6c945d36cb7bea8bd54ce3e19e04f39fec56274d42faf96018fd19db",
        "82a95b368b2f6be381f10228026ebf2ffffc7e0d80bf5580ee905adf29b26128"];
        post_fetch_method= [unpack, unpack, unpack, unpack]
    ))
    return nothing
end
