# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 16:35:09 2023

@author: ariel
"""
import multiprocessing


def f():  # no argument
    return 1

if __name__ == "__main__":
    pool = multiprocessing.Pool(2)
    result = pool.starmap(f, [() for _ in range(10)])
    print(result)