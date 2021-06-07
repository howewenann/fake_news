from multiprocessing import Pool

def parallelize_dataframe(df, func, ncores=4):
    df_split = np.array_split(df, ncores)
    pool = Pool(ncores)
    out = pd.concat(pool.map(func, df_split), ignore_index=True, sort=True)
    pool.close()
    pool.join()
    return out