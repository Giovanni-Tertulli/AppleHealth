#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from data_processor import import_process, fix_dates, display_metrics, display_macro_split, display_pct_change_GI

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[ ]:


df = import_process()
df, weekly_df = fix_dates(df)
display_metrics(weekly_df)
display_macro_split(weekly_df)
display_pct_change_GI(weekly_df)


# In[ ]:




