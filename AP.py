##########################################
# Asset pricing                          #
# Wu Jiaying                             #
# Date: Sep. 2023                        #
# Updated: Sept. 28th 2024               #
##########################################
# Note:
# The class is used for univariate and bivariate sorting, as well as Fama-MacBeth regression.
# The Compustat and CRSP blocks load accounting and stock price data, while the CCM block links Compustat and CRSP.
# The functions for Table 1, Table 2, and Table 3 correspond to univariate sorting, bivariate sorting, and Fama-MacBeth regression, respectively.
# The Interest Variables block is used to merge a unique dataset containing the variables aimed at seeking alpha. (Writing code to merge ccm_month_win with other dataset)
##########################################

# *** For the use of relevant options, please refer to the specific function descriptions. ***

## Call this package outside .py ##
# from AP import AP_table

######################################################################
# Compustat and CRSP data are downloaded from WRDS,
# while factors are obtained from the Kenneth R. French Data Library.
######################################################################

# AP=AP_table(data_path='',time_range=['20000101','20221231'],portfolio_month=6)
# AP.Compustat(file='WRDS/Compustat_2000_2022.csv')
# crsp_monthly=AP.CRSP(file_var='WRDS/CRSP_Monthly_2000_2022.csv',file_name='WRDS/NAME.csv',file_dl='WRDS/Delisted Stocks.csv')
# ccm_month_win=AP.CCM(file='WRDS/CCM_CRSP_Link_Table_CRSP.csv')
# factors=AP.F_F_Factors(file_FF5='F_F/F-F_5_Factors_monthly.csv',file_MOM='F_F/F-F_Momentum_monthly.CSV')
# main_month_win=AP.Main_var(file='') # Interest variables block

# For example:
# Variable: cosine_sim_tnic_1_trim_0.05
# Time period: [2011, 2017]	
# Quantile group: [[0, 0.3, 0.7, 1], ['L', 'M', 'H']]

### Univariate Sorting ###
# AP.Table1(future_return=1,sample_window=[2011, 2017],Table_Kind='Alpha',sort_var=['XXXport','cosine_sim_tnic_1_trim_0.05'],\
                #   q_t='Quantile',q_df='Self',sort_q_l=[[0, 0.3, 0.7, 1], ['L', 'M', 'H']],maxlags=20)

### Independent Bivariate Sorting ###
# cosine_sim_tnic_1_trim_0.05 and ln_market_value

# tab2=AP.Table2(future_return=1,sample_window=[2011, 2017]	,\
#             sort_var=['XXXport','cosine_sim_tnic_1_trim_0.05'],q_t='Quantile',q_df='Self',sort_q_l=[[0, 0.3, 0.7, 1], ['L', 'M', 'H']],maxlags=20,\
#                         second_sort_list=[['ln_market_value_port','ln_market_value']],\
#                         second_sort_q_t_df=[['Quantile','Self']],\
#                         second_sort_q_l=[[[0, 0.3, 0.7, 1], ['L', 'M', 'H']]])

### Fama-MacBeth ###
# # Add new control variables
# jp_new=AP.control()
# # Shift y_lag month of excess return as y variables.
# jp_table3=AP.Table3(y_lag=[1,2,3,6,9,12])
# control_var=['ln_market_value','ln_Assets','ln_firm_age','ln_PPE/Employees','RD/Assets',\
#                 'ROA','beme','Cash/Assets','Beta5Y_monthly','Momentum_monthly','Stock_volatility']
# AP.FamaMacBeth(time_series=['2011'+'0731','2018'+'0630'],\
#                          x=[],control=control_var)


##########################################
import os
import datetime
import pandas as pd
import numpy as np
import math

from collections import Counter

import matplotlib.pyplot as plt
from dateutil.relativedelta import *
from pandas.tseries.offsets import *

import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels import FamaMacBeth 
##########################################

class AP_table():
    def __init__(self,data_path,time_range=['19630101','19911231'],portfolio_month=6):
        self.data_path=data_path
        self.time_range=time_range
        self.portfolio_month=portfolio_month

        pass

    ###################
    # Compustat Block #
    ###################
    def Compustat(self,file,):

        comp=pd.read_csv(os.path.join(self.data_path,file),low_memory=False)

        # https://www.crsp.org/products/documentation/annual-data-industrial
        comp=comp[['GVKEY','cusip','datadate','at','pstkl','txditc','pstkrv','seq','pstk','datafmt','indfmt',\
                   'popsrc','consol','revt','cogs','xsga','xrd','ebit','ppent','oibdp','ceq','prcc_f','csho','ib','sic','loc',\
                    'emp','rdip','salepfc','salepfp','sale','ch','oiadp','txdb','ni','xint','dlc','dltt']]

        # at: Assets - Total
        # pstkl: Preferred Stock/Liquidating Value
        # txditc: Deferred Taxes and Investment Tax Credit
        # pstkrv: Preferred Stock/Redemption Value
        # seq: Stockholders' Equity - Total
        # pstk: Preferred/Preference Stock (Capital) - Total
        # SALE -- Sales/Turnover (Net) (SALE)
        # SALEPFC -- Pro Forma Net Sales - Current Year (SALEPFC)
        # SALEPFP -- Pro Forma Net Sales - Prior Year (SALEPFP)

        # DATAFMT='STD' and INDFMT='INDL' and CONSOL='C' and POPSRC='D' to retrieve the standardized (as opposed to re-stated data), 
        # consolidated (as opposed to pro-forma) data presented in the industrial format (as opposed to financial services format) 
        # for domestic companys (as opposed to international firms), i.e., the U.S. and Canadian firms.
        
        ###################### Conditions on data #####################
        comp=comp[(comp['indfmt']=='INDL')&(comp['datafmt']=='STD')&(comp['popsrc']=='D')&(comp['consol']=='C')]
        comp['year']=pd.to_datetime(comp['datadate']).dt.year
        comp=comp.drop_duplicates(subset=['GVKEY','year'],keep='last')
        

        # ROA: Return on assets (ROA) Earnings before extraordinary items divided by assets (ib/at)
        # Book-to-market Book value of common equity [ceq] divided by the market value of common equity [prcc_f x csho].
        # Operating profitability (Ball R, Gerakos J, Linnainmaa J T, et al. Accruals, cash flows, and operating profitability in the cross section of stock returns[J]. Journal of Financial Economics, 2016, 121(1): 28-45.)
        # ≡ Revenue (REVT)
        # − Cost of goods sold (COGS)
        # −Reported sales, general, and
        # administrative expenses (XSGA− XRD)
        
        # Keep the data with positive assets
        comp=comp[comp['at']>0]

        # Hiring Rate
        # HNt =Ht/ (0.5 × (Nt−1 + Nt))
        # Ht=Nt-Nt-1
        comp['HR']=comp.groupby('GVKEY')['emp'].transform(lambda x: (x-x.shift(1))/(0.5*(x.shift(1)+x)))


        # Future Profitability
        comp['ROA_fp']=comp['oiadp']/comp['at']

        comp['ROA']=comp['ib']/comp['at']

        # return on assets (ROA) measured as the ratio of net income plus interest expense to assets
        comp['ROA_ai']=(comp['ni']+comp['xint'])/comp['at']

        comp['Market_leverage']=(comp['dlc']+comp['dltt'])/(comp['csho']*comp['prcc_f'])

        comp['Book_to_market_2']=comp['ceq']/(comp['prcc_f']*comp['csho'])

        comp['INV']=comp.groupby('GVKEY',group_keys=False)['at'].apply(lambda x: (x-x.shift(1))/x.shift(1))
        comp['OP']=comp['revt']-comp['cogs']-comp['xsga']+comp['xrd']
        comp['OP/at']=comp['OP']/comp['at']

        comp['PPET']=comp['ppent']
        comp['PPET/at']=comp['ppent']/comp['at']

        comp['ln_Assets']=np.log(comp['at'])
        comp['ln_PPE/Employees']=np.log(comp['ppent']/comp['emp'])
        comp['RD/Assets']=comp['rdip']/comp['at']
        comp['RD/Sales']=comp['rdip']/comp['sale']
        comp['Sales_growth']=np.log(1+comp['salepfc']/comp['salepfp'])
        comp['Cash/Assets']=comp['ch']/comp['at']

        comp['ln_Makeup']=np.log(comp['cogs']/comp['sale'])


        # Replace the inf and -inf to nan while using log
        comp=comp.replace(-np.inf,np.nan)
        comp=comp.replace(np.inf,np.nan)

        comp=comp[(pd.to_datetime(comp.datadate)>=datetime.datetime.strptime('19590101','%Y%m%d'))]
        # Def the time series order of stock
        comp['count']=comp.groupby(['GVKEY']).cumcount()
        
        # Set time range
        comp=comp[(pd.to_datetime(comp.datadate)>=datetime.datetime.strptime(self.time_range[0],'%Y%m%d'))&(pd.to_datetime(comp.datadate)<=datetime.datetime.strptime(self.time_range[1],'%Y%m%d'))]

        ####################### Calculate book equity (be) #####################
        # Book Equity. Book equity is constructed from Compustat data or collected from the Moody’s Industrial, Financial, and Utilities manuals. 
        # BE is the book value of stockholders’ equity, plus balance sheet deferred taxes and investment tax credit (if available), minus the book value of preferred stock. 
        # Depending on availability, we use the redemption, liquidation, or par value (in that order) to estimate the book value of preferred stock. 
        
        # create preferrerd stock
        comp['ps']=np.where(comp['pstkrv'].isnull(), comp['pstkl'], comp['pstkrv'])
        comp['ps']=np.where(comp['ps'].isnull(),comp['pstk'], comp['ps'])
        comp['ps']=np.where(comp['ps'].isnull(),0,comp['ps'])
        comp['txditc']=comp['txditc'].fillna(0)

        # create book equity
        # Note that we set negative or zero equity to missing which is a common practice when working with book-to-market ratios.
        # (see Fama and French 1992 for details)

        comp['be']=comp['seq']+comp['txditc']-comp['ps']
        comp['be']=np.where(comp['be']>0, comp['be'], np.nan)

        # comp['OP']=comp['profit']/comp['be']

        # Tobin's Q
        # (a.prcc_f*a.csho+a.at-a.ceq-coalesce(a.txdb,0))/a.at as TobinQ : https://zhuanlan.zhihu.com/p/537813671
        comp['txdb'] = comp['txdb'].fillna(0)

        comp['TobinQ'] = ((comp['prcc_f'] * comp['csho']) + comp['at'] - comp['ceq'] - comp['txdb']) / comp['at']

        # number of years in Compustat
        comp=comp.sort_values(by=['GVKEY','datadate'])


        # The OP for June of year t is calculated as annually revenues minus cost of goods sold,
        # interest expense, and selling, general, and administrative expenses divided by book
        # equity for the last fiscal year end in t − 1. The investment portfolios are formed on the
        # change in total assets from the fiscal year ending in year t − 2 to the fiscal year ending
        # in t − 1, divided by t − 2 total assets at the end of each June.

        # We calculate operating profitability by following the computations in Ball, Gerakos, Linnainmaa and Nikolaev (2015):
        # sales minus cost of goods sold minus sales, general, and administrative expenses (excluding research and development expenditures).

        comp=comp[['GVKEY','cusip','datadate','year','be','count','INV','OP','ROA','ROA_fp','PPET','PPET/at','Book_to_market_2','OP/at','sic','loc',\
                   'ln_Assets','ln_PPE/Employees','RD/Assets','Sales_growth','Cash/Assets','sale','salepfc','salepfp','TobinQ','RD/Sales','ln_Makeup',\
                    'ROA_ai','Market_leverage','HR','emp']]
        
        self.financial_var=['INV','OP','ROA','ROA_fp','PPET','PPET/at','Book_to_market_2','OP/at','ln_Assets',\
                            'ln_PPE/Employees','RD/Assets','Sales_growth','Cash/Assets','sale','salepfc','salepfp',\
                                'TobinQ','RD/Sales','ln_Makeup','ROA_ai','Market_leverage','HR','emp']
        comp=comp.rename(columns={'GVKEY':'gvkey'})

        self.comp=comp
        return comp

    ###################
    # CRSP Block      #
    ###################
    def CRSP(self,file_var,file_name,file_dl):
        # Load CRSP variables
        crsp_m = pd.read_csv(os.path.join(self.data_path,file_var),low_memory=False)
        crsp_m=crsp_m[['PERMNO','PERMCO','TICKER','date','RET','RETX','SHROUT','PRC','NCUSIP','CUSIP']]

        # permco	double	CRSP Permanent Company Number (permco)
        # ret	double	Holding Period Return (ret)
        # retx	double	Holding Period Return without Dividends (retx)
        # shrout	double	Number of Shares Outstanding (shrout)
        # prc	double	Price (prc)

        # Load CRSP NAME
        crsp_name = pd.read_csv(os.path.join(self.data_path,file_name))
        crsp_name=crsp_name[['PERMNO','DATE','NAMEENDT','SHRCD','EXCHCD']]
        crsp_name=crsp_name[crsp_name.EXCHCD.isin([1,2,3])]
        crsp_name.columns=['PERMNO','NAMEDT','NAMEENDT','SHRCD','EXCHCD']

        # shrcd	double	Share Code (shrcd)
        # exchcd	double	Exchange Code (exchcd)
        # nameendt	date	Names Ending Date (nameendt)

        ########################################## Conditions on data ###############################
        crsp_m1=pd.merge(crsp_m,crsp_name,on=['PERMNO'],how='left')
        crsp_m1=crsp_m1[(crsp_m1['date']>=crsp_m1['NAMEDT'])&(crsp_m1['date']<=crsp_m1['NAMEENDT'])]

        crsp_m1=crsp_m1[(pd.to_datetime(crsp_m1.date)>=datetime.datetime.strptime('19590101','%Y%m%d'))]
        # &(pd.to_datetime(crsp_m1.date)<=datetime.datetime.strptime('20171231', '%Y%m%d'))

        crsp_m1["exchange"] = crsp_m1["EXCHCD"].apply(assign_exchange)
        crsp_m1.exchange.value_counts()

        # change variable format to int
        crsp_m1[['PERMCO','PERMNO','SHRCD','EXCHCD']]=crsp_m1[['PERMCO','PERMNO','SHRCD','EXCHCD']].astype('Int64')

        # Line up date to be end of month
        crsp_m1['jdate']=pd.to_datetime(crsp_m1['date'])+MonthEnd(0)

        # add delisting return
        dlret = pd.read_csv(os.path.join(self.data_path,file_dl))
        dlret=dlret[['PERMNO','DLRET','DLSTDT']]
        dlret.PERMNO=dlret.PERMNO.astype(int)
        
        # Line up date to be end of month
        dlret['jdate']=pd.to_datetime(dlret['DLSTDT'])+MonthEnd(0)

        crsp = pd.merge(crsp_m1, dlret, how='left',on=['PERMNO','jdate'])

        crsp['DLRET']=crsp['DLRET'].fillna(0)

        crsp['RET']=crsp['RET'].fillna(0)

        ######################################### Cleaning the data #####################################
        # There are some observations with a str form of "RET" "RETX" "DLRET".

        crsp['RET']=crsp['RET'].apply(convert_currency)
        crsp['RETX']=crsp['RETX'].apply(convert_currency)

        # Find the str part data
        crsp_str=crsp[(crsp['RET'].apply(lambda x: isinstance(x,str)))|(crsp['RETX'].apply(lambda x: isinstance(x,str)))|(crsp['DLRET'].apply(lambda x: isinstance(x,str)))]
        crsp_now=crsp[(~crsp['RET'].apply(lambda x: isinstance(x,str)))&(~crsp['RETX'].apply(lambda x: isinstance(x,str)))&(~crsp['DLRET'].apply(lambda x: isinstance(x,str)))]
        print('The percentage of error observations:',np.round(crsp_str.shape[0]/crsp.shape[0]*100,2),'%')
        crsp_now = crsp_now.astype({"RET":'float', "RETX":'float'})
        
        # retadj factors in the delisting returns
        crsp_now['retadj']=(1+crsp_now['RET'])*(1+crsp_now['DLRET'])-1

        # **We found that some observations' SHROUT is zero which will cause the me to be zero and beme to be inf. Thus replacing with nan.**
        crsp_now['SHROUT'].replace(0,np.nan,inplace=True)

        ##################################### Calculate Some Indicators #####################################
        
        # calculate market equity
        crsp_now['me']=crsp_now['PRC'].abs()*crsp_now['SHROUT'] 
        crsp_now=crsp_now.drop(['DLRET','DLSTDT','PRC','SHROUT'], axis=1)
        crsp_now=crsp_now.sort_values(by=['jdate','PERMCO','me'])

        # Deal with the company which had not unique datas in a given date 
        # Notice: (If there are several values in one day of one particular permco stock, keep the one observation with maximum me.)

        ### Aggregate Market Cap —— For Value-Weighted ###
        # sum of me across different permno belonging to same permco a given date
        crsp_summe = crsp_now.groupby(['jdate','PERMCO'])['me'].sum().reset_index()
        # largest mktcap within a permco/date
        crsp_maxme = crsp_now.groupby(['jdate','PERMCO'])['me'].max().reset_index()
        # join by jdate/maxme to find the permno
        crsp1=pd.merge(crsp_now, crsp_maxme, how='inner', on=['jdate','PERMCO','me'])
        # drop me column and replace with the sum me
        crsp1=crsp1.drop(['me'], axis=1)
        # join with sum of me to get the correct market cap info
        crsp2=pd.merge(crsp1, crsp_summe, how='inner', on=['jdate','PERMCO'])
        # sort by permno and date and also drop duplicates
        # The same me observations of a given data and permco would keep and now be dropped.
        crsp2=crsp2.sort_values(by=['PERMNO','jdate']).drop_duplicates()

        # **Using pandas.sum(), the nan columns will return 0. Thus, we need to replace.**
        crsp2['me']=crsp2['me'].replace(0,np.nan)


        # keep December market cap : Use for computing its book-to-market, leverage, and earnings-price ratios
        crsp2['year']=crsp2['jdate'].dt.year
        crsp2['month']=crsp2['jdate'].dt.month
        decme=crsp2[crsp2['month']==12]
        decme=decme[['PERMNO','date','jdate','me','year']].rename(columns={'me':'dec_me'})

        ### Window_month set back self.portfolio_month month
        crsp2['ffdate']=crsp2['jdate']+MonthEnd(-self.portfolio_month)
        crsp2['ffyear']=crsp2['ffdate'].dt.year
        crsp2['ffmonth']=crsp2['ffdate'].dt.month

        crsp2['1+retx']=1+crsp2['RETX']
        crsp2=crsp2.sort_values(by=['PERMNO','date'])

        # cumret by stock
        crsp2['cumretx']=crsp2.groupby(['PERMNO','ffyear'])['1+retx'].cumprod()

        # lag cumret
        crsp2['lcumretx']=crsp2.groupby(['PERMNO'])['cumretx'].shift(1)

        # lag market cap
        crsp2['lme']=crsp2.groupby(['PERMNO'])['me'].shift(1)

        # if first permno then use me/(1+retx) to replace the missing value
        crsp2['count']=crsp2.groupby(['PERMNO']).cumcount()
        crsp2['lme']=np.where(crsp2['count']==0, crsp2['me']/crsp2['1+retx'], crsp2['lme'])

        # baseline me
        mebase=crsp2[crsp2['ffmonth']==1][['PERMNO','ffyear', 'lme']].rename(columns={'lme':'mebase'})

        # merge result back together
        crsp_monthly=pd.merge(crsp2, mebase, how='left', on=['PERMNO','ffyear'])

        # Calculate firm age
        crsp_monthly['ln_firm_age']=np.log(crsp_monthly.groupby(['PERMNO']).cumcount()+1)
        crsp_monthly['firm_age']=crsp_monthly.groupby(['PERMNO']).cumcount()+1

        # Set time range
        crsp_monthly=crsp_monthly[(pd.to_datetime(crsp_monthly.date)>=datetime.datetime.strptime(self.time_range[0],'%Y%m%d'))\
                                  &(pd.to_datetime(crsp_monthly.date)<=datetime.datetime.strptime(self.time_range[1],'%Y%m%d'))]

        # Notice: (Using the cumret is for adjusting market equity to the price level at base month(The first month of a portfolio))

        crsp_monthly['wt']=np.where(crsp_monthly['ffmonth']==1, crsp_monthly['lme'], crsp_monthly['mebase']*crsp_monthly['lcumretx'])

        crsp_monthly['retadj']=crsp_monthly['retadj']*100

        self.crsp_monthly=crsp_monthly

        # ************************* year+1
        decme['year']=decme['year']+1
        decme=decme[['PERMNO','year','dec_me']]

        # Info as of The Window Month
        crsp_monthly_begin = crsp_monthly[crsp_monthly['month']==self.portfolio_month]

        crsp_month_win = pd.merge(crsp_monthly_begin, decme, how='inner', on=['PERMNO','year'])
        crsp_month_win=crsp_month_win[['PERMNO','TICKER','date','NCUSIP','CUSIP','jdate', 'SHRCD','EXCHCD','retadj','me','wt','cumretx','mebase','lme','dec_me','ln_firm_age','firm_age']]
        crsp_month_win['ln_market_value']=np.log(crsp_month_win['me'])

        crsp_month_win=crsp_month_win.sort_values(by=['PERMNO','jdate']).drop_duplicates()

        self.crsp_month_win=crsp_month_win

        return crsp_monthly

    #######################
    # CCM Block           #
    #######################
    def CCM(self,file):
        ccm=pd.read_csv(os.path.join(self.data_path,file),low_memory=False)

        ccm=ccm[['gvkey','LPERMNO','LINKTYPE','LINKPRIM','LINKDT','LINKENDDT']]
        ccm.columns=['gvkey','PERMNO','LINKTYPE','LINKPRIM','linkdt','linkenddt']
        ccm['PERMNO']=ccm['PERMNO'].astype('Int64')

        # SQL
        ccm=ccm[(ccm['LINKTYPE'].str[:1]=='L')&(ccm['LINKPRIM'].isin(['C','P']))]

        # if linkenddt is missing then set to today date
        ccm['linkenddt']=ccm['linkenddt'].mask(ccm['linkenddt']=='E',pd.to_datetime('today').strftime('%Y/%m/%d'))

        ccm1=pd.merge(self.comp,ccm,how='left',on=['gvkey'])
        ccm1['yearend']=pd.to_datetime(ccm1['datadate'])+YearEnd(0)
        ccm1['jdate']=pd.to_datetime(ccm1['yearend'])+MonthEnd(self.portfolio_month)

        ccm1['linkdt']=pd.to_datetime(ccm1['linkdt'])
        ccm1['linkenddt']=pd.to_datetime(ccm1['linkenddt'])

        # set link date bounds
        ccm2=ccm1[(ccm1['jdate']>=ccm1['linkdt'])&(ccm1['jdate']<=ccm1['linkenddt'])]
        ccm2=ccm2[['gvkey','cusip','PERMNO','datadate','yearend', 'jdate','be', 'count','sic']+self.financial_var]

        ccm2['year']=ccm2['jdate'].dt.year.copy()

        # Keep the laterest data
        ccm2['datadate']=pd.to_datetime(ccm2['datadate'])

        ccm2=ccm2.sort_values(['PERMNO', 'jdate','datadate'],ascending=False)

        ccm2=ccm2.drop_duplicates(subset=['PERMNO', 'jdate'],keep='first')

        # link comp and crsp
        ccm_month_win=pd.merge(self.crsp_month_win, ccm2, how='inner', on=['PERMNO', 'jdate'])
        ccm_month_win['beme']=ccm_month_win['be']*1000/ccm_month_win['dec_me']

        self.ccm_month_win=ccm_month_win

        self.financial_var=self.financial_var+['ln_firm_age','firm_age','ln_market_value']

        return ccm_month_win
    
    ##############################
    #  Interest Variables Block  #
    ##############################
    def Main_var(self,file=''):
        pass


    ### Winsorize quantile ******
    def winsorize_quatile(self,variable=['low_tech_jobs_num_winsorized','high_tech_jobs_num_winsorized'],upper_quantile=0.95,lower_quantile=0.05):
        # Winsorize quantile
        for v in variable:
            self.ccm_month_win[v+'_winsorized']=self.ccm_month_win.groupby(['year'],group_keys=False)[v]\
                .apply(lambda x: x.clip(upper=x.quantile(upper_quantile),lower=x.quantile(lower_quantile)))
        return self.ccm_month_win
    
    ############################
    # Load Fama French Factors #
    ############################
    def F_F_Factors(self,file_FF5='F_F/F-F_5_Factors_monthly.csv',file_MOM='F_F/F-F_Momentum_monthly.CSV'):
        # Fama-French 5 factors
        FF5=pd.read_csv(file_FF5)
        FF5['jdate']=FF5['jdate'].apply(lambda x: datetime.datetime.strptime(str(x),'%Y%m'))+MonthEnd()

        # Momentum factor
        MOM=pd.read_csv(file_MOM)
        MOM['jdate']=MOM['jdate'].apply(lambda x: datetime.datetime.strptime(str(x),'%Y%m'))+MonthEnd()
        MOM.columns=['jdate','MOM']

        factors=pd.merge(FF5,MOM,on='jdate',how='inner')
        factors=factors.rename(columns={'Mkt-RF':'Mkt_RF'})

        self.factors=factors
        return factors

    # Independent Sorting
    # q_data is the data for calculating portfolio breaking point. If "NYSE" means use the nyse stork data,
    # or "Self" means use all the correspondent data, otherwise, you should offer the data for calculating.
    def Sorting(self,x,sort_variable=['szport','me'],q_data='NYSE',q_tp='Quantile',q_point=[0,0.5,1],q_label=['S','B']):
        
        # select NYSE stocks for bucket breakdown
        # exchcd = 1 and positive beme and positive me and shrcd in (10,11) and at least 2 years in comp
        if q_data=='NYSE':
            nyse=x[(x['EXCHCD']==1) & (x['beme']>0) & (x['me']>0) & \
                        (x['count']>=1) & ((x['SHRCD']==10) | (x['SHRCD']==11))]
        
            x.loc[x['count']>=1,sort_variable[0]]=x[x['count']>=1]\
                .groupby(['jdate'],group_keys=False)[sort_variable[1]].apply(qfunc,q_data=nyse,sort_col=sort_variable[1],q_type=q_tp,q=q_point,label=q_label)
            
            x.loc[(x['count']<1),sort_variable[0]]=np.nan

        elif q_data=='Self':
            x.loc[x['count']>=1,sort_variable[0]]=x[x['count']>=1]\
                .groupby(['jdate'],group_keys=False)[sort_variable[1]].apply(qfunc,q_data=x[x['count']>=1],sort_col=sort_variable[1],q_type=q_tp,q=q_point,label=q_label)
            
            x.loc[(x['count']<1),sort_variable[0]]=np.nan
        
        # if not use nyse for calculating breaking point, you need to offer the data for calculating breaking point.
        else:
            x.loc[x['count']>=1,sort_variable[0]]=x[x['count']>=1]\
                .groupby(['jdate'],group_keys=False)[sort_variable[1]].apply(qfunc,q_data=q_data,sort_col=sort_variable[1],q_type=q_tp,q=q_point,label=q_label)
            
            x.loc[(x['count']<1),sort_variable[0]]=np.nan
        
        return x
    
    # Univariate Sorting: Equal-weighted & Value-weighted Excess return, CAPM Alpha, FF3, FFC, FF5
    # sample_window: The portfolio time series for regression.
    # Table_Kind: "Alpha" - Only output the alpha coefficients. "Alpha + Beta" will output alpha and beta in the models.
    # sort_var: sort_var[0] - The name of the sorting portfolio. sort_var[1] - sorting variable.
    # q_t: If "Quantile", it will divide the groups by quantiles, or if "Censore", it will use the first one as a single value group.
    # q_df: The data for calculating every time every portfolio's breaking points.
    # sort_q_l: Sorting quantiles and label
    # maxlags: Newey-west adjust
    def Table1(self,future_return=1,sample_window=[1963,2023],Table_Kind='Alpha',sort_var=['XXXport','VAR'],q_t='Quantile',q_df='NYSE',sort_q_l=[[0,0.3,0.7,1],['L','M','H']],maxlags=1):
        
        # If we sort one variable from financial variables, we should remove it to avoid duplicated calculations later.
        self.financial_var1 = list(filter(lambda x: x != sort_var[1], self.financial_var))

        ccm_month_win=self.ccm_month_win.copy()

        ccm_month_win=ccm_month_win[(ccm_month_win['year']>=sample_window[0])&(ccm_month_win['year']<=sample_window[1])]

        # Dropna of sort_var
        ccm_month_win.dropna(subset=[sort_var[1]],inplace=True)

        # Univariate sorting
        try:
            ccm_month_win=AP_table.Sorting(self,ccm_month_win,sort_variable=sort_var,q_tp=q_t,q_data=q_df,q_point=sort_q_l[0],q_label=sort_q_l[1])
        except:
            print('Some timings could not figure out the exact quantile distribution breaking points, which means that some datas would be drop as nan.')
            # Generate empty set
            Table1_final,Table_stat=pd.DataFrame(np.full((1,len(sort_q_l[1])+1),fill_value=np.nan),columns=sort_q_l[1]+['H_L']),\
                pd.DataFrame(np.full((1,len(sort_q_l[1])+1),fill_value=np.nan),columns=sort_q_l[1]+['H_L'])
            return Table1_final,Table_stat
    
        
        # store portfolio assignment as of Given Month
        month_win=ccm_month_win[['gvkey','PERMNO','date','jdate','beme']+self.financial_var1+sort_var]
        o=month_win.copy()
        o['ffyear']=month_win['jdate'].dt.year

        # merge back with monthly records
        crsp_monthly = self.crsp_monthly[['date','PERMNO','SHRCD','EXCHCD','retadj','me','wt','cumretx','ffyear','jdate']]
        ccm_monthly=pd.merge(crsp_monthly, 
                o[['PERMNO','ffyear','beme']+self.financial_var1+sort_var], how='left', on=['PERMNO','ffyear'])

        # keeping only records that meet the criteria
        ccm_monthly=ccm_monthly[(ccm_monthly['wt']>0) & (ccm_monthly[sort_var[0]].notna()) & 
                ((ccm_monthly['SHRCD']==10) | (ccm_monthly['SHRCD']==11))]
        
        ccm_monthly['year']=ccm_monthly['jdate'].dt.year.copy()

        # Set time range
        ccm_monthly=ccm_monthly[(pd.to_datetime(ccm_monthly.jdate)>=datetime.datetime.strptime(str(sample_window[0])+'0731','%Y%m%d'))&\
                  (pd.to_datetime(ccm_monthly.jdate)<=datetime.datetime.strptime(str(sample_window[1]+1)+'0630','%Y%m%d'))]

        # Statistics
        ccm_monthly['Firm Number']=1
        ccm_monthly[sort_var[0]]=ccm_monthly[sort_var[0]].astype(pd.CategoricalDtype(sort_q_l[1],ordered=True))
        
        # Firm number: Average jdate (Time) firm coount.
        # t1=ccm_monthly.groupby([sort_var[0]])['PERMNO'].apply(lambda x: x.unique().shape[0]).to_frame().rename(columns={0: 'Firm Number'}).T
        t1=ccm_monthly.groupby([sort_var[0],'jdate'])[['Firm Number']].sum().reset_index()\
            [[sort_var[0],'Firm Number']].groupby([sort_var[0]]).mean().T
        
        ccm_monthly['me']=ccm_monthly['me']/1000000 # dollars to million dollars

        t2=ccm_monthly.groupby([sort_var[0],'jdate'])[[sort_var[1],'me','beme']+self.financial_var1].mean().reset_index()\
            [[sort_var[0],sort_var[1],'me','beme']+self.financial_var1].groupby([sort_var[0]]).mean().T

        # Average Firm Number of portfolio across year. Average me beme in portfolio and average year
        Table_stat=pd.concat([t1,t2])
        index1=['Market value' if i == 'me' else i for i in Table_stat.index ]
        index2=['Book-to-market' if i == 'beme' else i for i in index1]
        
        Table_stat.index=index2

        # Equal-weighted & Value-weighted return rate.
        vw_ret=ccm_monthly.groupby(['jdate',sort_var[0]],group_keys=True).apply(wavg, 'retadj','wt').reset_index().rename(columns={0: 'vwret'})\
            .pivot(index=['jdate'], columns=[sort_var[0]], values='vwret')

        ew_ret=ccm_monthly.groupby(['jdate',sort_var[0]],group_keys=True)['retadj'].mean().reset_index().rename(columns={'retadj': 'ewret'})\
            .pivot(index=['jdate'], columns=[sort_var[0]], values='ewret')
        
        # Rename the variables
        vw_ret.columns=[i+'_vw' for i in sort_q_l[1]]
        ew_ret.columns=[i+'_ew' for i in sort_q_l[1]]
        
        # Calculate the H-L group
        vw_ret['H_L_vw']=vw_ret.iloc[:,-1]-vw_ret.iloc[:,0]
        ew_ret['H_L_ew']=ew_ret.iloc[:,-1]-ew_ret.iloc[:,0]

        # Combine with the factors time series.
        time_data=pd.merge(vw_ret,ew_ret,on='jdate',how='inner')

        # Calculate excess return
        RF_data=pd.merge(time_data.reset_index(),self.factors[['jdate','RF']],on='jdate',how='inner').set_index('jdate')['RF']
        time_data=time_data.sub(RF_data,axis=0)

        # Future 1 month return rate
        ret_future1=time_data.reset_index().copy()

        ret_future1['jdate']=time_data.reset_index()['jdate'].shift(future_return)
        ret_future1.dropna(inplace=True)

        # Regression variables
        y_list=ret_future1.columns.tolist()[1:]

        # Future 1 month return rate combine factors
        ret_future1_fac=pd.merge(ret_future1,self.factors,on='jdate',how='inner')
        
        if Table_Kind == "Alpha":
            # Alpha_list:
            # res_list=[['Excess Return','1'],['CAMP alpha','1 + Mkt_RF'],['FF3 alpha','1 + Mkt_RF + SMB + HML'],\
            #         ['FFC alpha','1 + Mkt_RF + SMB + HML + MOM'],['FF5 alpha','1 + Mkt_RF + SMB + HML + RMW + CMA']]
            # res_type=['Expected Average Excess Return','CAMP','FF3','FFC','FF5']
            res_list=[['Excess Return','1'],['CAMP alpha','1 + Mkt_RF'],\
                    ['FFC alpha','1 + Mkt_RF + SMB + HML + MOM'],['FF5 alpha','1 + Mkt_RF + SMB + HML + RMW + CMA']]
            res_type=['Expected Average Excess Return','CAMP','FFC','FF5']

            res_main=[]

            # (Risk-adjusted return)
            ra_ret=ret_future1_fac[['jdate']]

            # Regression and Store results
            for i in y_list:
                for j0,j in enumerate(res_list):
                    # Time Series Regression
                    res=smf.ols(i + ' ~ ' + j[1],data=ret_future1_fac).fit(cov_type='HAC',cov_kwds={'maxlags':20})
                    res_main.extend([[i,j[0],res.params[0],res.bse[0],res.tvalues[0],res.pvalues[0]]])

                    if j[0]!='Excess Return':
                        # Residual+Intercept (Risk-adjusted return)
                        ra_ret1=pd.DataFrame(res.resid+res.params[0],columns=[tuple([i,res_type[j0]])])

                        ra_ret=pd.concat([ra_ret,ra_ret1],axis=1)

            ra_ret=ra_ret.set_index(['jdate'])
            ra_ret.columns=pd.MultiIndex.from_tuples(ra_ret.columns)
            
            table1=pd.DataFrame(res_main,columns=['Y','Form','coef','std','tvalue','pvalue'])
            table1['Kind']=table1['Y'].str[-2:]
            table1['Portfolio_Group']=table1['Y'].str[:-3]
            
            # Set order
            order = pd.CategoricalDtype(sort_q_l[1]+['H_L'],ordered=True)
            order1 = pd.CategoricalDtype([k[0] for k in res_list],ordered=True)

            table1['Portfolio_Group']=table1['Portfolio_Group'].astype(order)
            table1['Form']=table1['Form'].astype(order1)

            Table1_coef=table1.pivot_table(index=['Form','Kind'],columns=['Portfolio_Group'],values=['coef'])
            Table1_p=table1.pivot_table(index=['Form','Kind'],columns=['Portfolio_Group'],values=['pvalue'])
            Table1_std=table1.pivot_table(index=['Form','Kind'],columns=['Portfolio_Group'],values=['std'])

        elif Table_Kind == "Alpha + Beta":
            # Alpha_Beta_list:
            # res_list=[['Average Excess Return',['Return'],'1'],['CAMP',['CAMP alpha','CAMP Mkt_RF Beta'],'1 + Mkt_RF'],\
            #         ['FF3',['FF3 alpha','FF3 Mkt_RF Beta','FF3 SMB Beta','FF3 HML Beta'],'1 + Mkt_RF + SMB + HML'],\
            #             ['FFC',['FFC alpha','FFC Mkt_RF Beta','FFC SMB Beta','FFC HML Beta','FFC MOM Beta'],'1 + Mkt_RF + SMB + HML + MOM'],\
            #             ['FF5',['FF5 alpha','FF5 Mkt_RF Beta','FF5 SMB Beta','FF5 HML Beta','FF5 RMW Beta','FF5 CMA Beta'],'1 + Mkt_RF + SMB + HML + RMW + CMA']]
            res_list=[['Average Excess Return',['Excess Return'],'1'],['CAMP',['CAMP alpha','CAMP Mkt_RF Beta'],'1 + Mkt_RF'],\
                        ['FFC',['FFC alpha','FFC Mkt_RF Beta','FFC SMB Beta','FFC HML Beta','FFC MOM Beta'],'1 + Mkt_RF + SMB + HML + MOM'],\
                        ['FF5',['FF5 alpha','FF5 Mkt_RF Beta','FF5 SMB Beta','FF5 HML Beta','FF5 RMW Beta','FF5 CMA Beta'],'1 + Mkt_RF + SMB + HML + RMW + CMA']]

            res_main=[]

            # Regression and Store results
            for i in y_list:
                for j0,j in enumerate(res_list):
                    # Time Series Regression
                    res=smf.ols(i + ' ~ ' + j[1],data=ret_future1_fac).fit(cov_type='HAC',cov_kwds={'maxlags':20})
                    res_main.extend([[i,j[0],res.params[0],res.bse[0],res.tvalues[0],res.pvalues[0]]])

            
            table1=pd.DataFrame(res_main,columns=['Y','Form','Var','coef','std','tvalue','pvalue'])
            table1['Kind']=table1['Y'].str[-2:]
            table1['Portfolio_Group']=table1['Y'].str[:-3]
            
            # Set order
            order = pd.CategoricalDtype(sort_q_l[1]+['H_L'],ordered=True)
            order1 = pd.CategoricalDtype([k[0] for k in res_list],ordered=True)
            order2 = pd.CategoricalDtype([t for k in res_list for t in k[1]],ordered=True)

            table1['Portfolio_Group']=table1['Portfolio_Group'].astype(order)
            table1['Form']=table1['Form'].astype(order1)
            table1['Var']=table1['Var'].astype(order2)

            Table1_coef=table1.pivot_table(index=['Form','Kind','Var'],columns=['Portfolio_Group'],values=['coef'])
            Table1_p=table1.pivot_table(index=['Form','Kind','Var'],columns=['Portfolio_Group'],values=['pvalue'])
            Table1_std=table1.pivot_table(index=['Form','Kind','Var'],columns=['Portfolio_Group'],values=['std'])

        # Set pvalue stars ***
        Table1_p=pvalue_func(Table1_p)

        Table1_coef=Table1_coef.T.reset_index(level=[0],drop=True).T
        Table1_p=Table1_p.T.reset_index(level=[0],drop=True).T
        Table1_std=Table1_std.T.reset_index(level=[0],drop=True).T

        # Combine statistics
        for e in range(Table1_coef.shape[0]):
            l=(Table1_coef.iloc[e].round(3).astype(str)+''+Table1_p.iloc[e].astype(str)+'('+Table1_std.iloc[e].round(3).astype(str)+')').to_frame()
            
            if e==0:
                l1=l.copy()
            else:
                # print(l,l1)
                l1=l1.join(l)

        Table1_final=l1.T

        notice='Notice: coef(p-value)(std): *p<.1; **p<.05; ***p<.01'

        # Store the return time series
        self.return_time_series=time_data

        return Table1_final,Table_stat,notice, self.return_time_series,ra_ret
        
    # Independent Bivariate Sorting
    # Equal-weighted & Value-weighted portfolios high-low average excess return and FF5 alpha.
    def Table2(self,future_return=1,sample_window=[1963,2023],\
               sort_var=['XXXport','VAR'],q_t='Quantile',q_df='NYSE',sort_q_l=[[0,0.3,0.7,1],['L','M','H']],maxlags=12,\
               second_sort_list=[['XXX1port','VAR'],['XXX2port','VAR'],['XXX3port','VAR']],\
               second_sort_q_t_df=[['Quantile','Self'],['Quantile','Self'],['Quantile','Self']],\
               second_sort_q_l=[[[0,0.5,1],['L','H']],[[0,0.5,1],['L','H']],[[0,0.5,1],['L','H']]]):
        # If we sort one variable from financial variables, we should remove it to avoid duplicated calculations later.
        self.financial_var1 = list(filter(lambda x: x != sort_var[1], self.financial_var))

        ccm_month_win=self.ccm_month_win.copy()
        # Time selection
        ccm_month_win=ccm_month_win[(ccm_month_win['year']>=sample_window[0])&(ccm_month_win['year']<=sample_window[1])]

        # Dropna of sort_var
        ccm_month_win.dropna(subset=[sort_var[1]],inplace=True)

        # Bivariate

        try:
            ccm_month_win=AP_table.Sorting(self,ccm_month_win,sort_variable=sort_var,q_tp=q_t,q_data=q_df,q_point=sort_q_l[0],q_label=sort_q_l[1])
        except:
            Table2_final=pd.DataFrame()
            return Table2_final

        # Second Sorting
        second_var=[]
        for g in range(len(second_sort_list)):
            try:
                ccm_month_win=AP_table.Sorting(self,ccm_month_win,sort_variable=second_sort_list[g],\
                                            q_tp=second_sort_q_t_df[g][0],q_data=second_sort_q_t_df[g][1],q_point=second_sort_q_l[g][0],q_label=second_sort_q_l[g][1])
                second_var.append(second_sort_list[g][0])
            except:
                Table2_final=pd.DataFrame()
                return Table2_final

        # store portfolio assignment as of Given Month
        month_win=ccm_month_win[['PERMNO','date','jdate','beme']+self.financial_var1+sort_var+second_var]
        o=month_win.copy()
        o['ffyear']=month_win['jdate'].dt.year

        # merge back with monthly records
        crsp_monthly = self.crsp_monthly[['date','PERMNO','SHRCD','EXCHCD','retadj','me','wt','cumretx','ffyear','jdate']]
        ccm_monthly=pd.merge(crsp_monthly, 
                o[['PERMNO','ffyear','beme']+self.financial_var1+sort_var+second_var], how='left', on=['PERMNO','ffyear'])

        # keeping only records that meet the criteria
        ccm_monthly=ccm_monthly[(ccm_monthly['wt']>0) & (ccm_monthly[sort_var[0]].notna()) & 
                ((ccm_monthly['SHRCD']==10) | (ccm_monthly['SHRCD']==11))]

        ccm_monthly['year']=ccm_monthly['jdate'].dt.year.copy()

        # Set time range
        ccm_monthly=ccm_monthly[(pd.to_datetime(ccm_monthly.jdate)>=datetime.datetime.strptime(str(sample_window[0])+'0731','%Y%m%d'))&\
                  (pd.to_datetime(ccm_monthly.jdate)<=datetime.datetime.strptime(str(sample_window[1]+1)+'0630','%Y%m%d'))]
        
        # Bivariate
        for g in range(len(second_sort_list)):
            # Equal-weighted & Value-weighted return rate.
            vw_ret=ccm_monthly.groupby(['jdate',sort_var[0],second_var[g]],group_keys=True).apply(wavg, 'retadj','wt').reset_index().rename(columns={0: 'vwret'})\
                .pivot(index=['jdate'], columns=[sort_var[0],second_var[g]], values='vwret')

            ew_ret=ccm_monthly.groupby(['jdate',sort_var[0],second_var[g]],group_keys=True)['retadj'].mean().reset_index().rename(columns={'retadj': 'ewret'})\
                .pivot(index=['jdate'], columns=[sort_var[0],second_var[g]], values='ewret')
            # print(ew_ret,vw_ret)
            # H-L group
            vw_ret=pd.DataFrame((vw_ret[sort_q_l[1][-1]].to_numpy()-vw_ret[sort_q_l[1][0]].to_numpy())[:,[0,-1]],index=vw_ret.index,columns=['H_L_vw_L','H_L_vw_H'])
            ew_ret=pd.DataFrame((ew_ret[sort_q_l[1][-1]].to_numpy()-ew_ret[sort_q_l[1][0]].to_numpy())[:,[0,-1]],index=ew_ret.index,columns=['H_L_ew_L','H_L_ew_H'])

            # Combine with the factors time series.
            time_data=pd.merge(vw_ret,ew_ret,on='jdate',how='inner')

            # Calculate excess return
            RF_data=pd.merge(time_data.reset_index(),self.factors[['jdate','RF']],on='jdate',how='inner').set_index('jdate')['RF']
            time_data=time_data.sub(RF_data,axis=0)

            # Future 1 month return rate
            ret_future1=time_data.reset_index().copy()
            ret_future1['jdate']=time_data.reset_index()['jdate'].shift(future_return)
            ret_future1.dropna(inplace=True)

            # Regression variables
            y_list=ret_future1.columns.tolist()[1:]

            # Future 1 month return rate combine factors
            ret_future1_fac=pd.merge(ret_future1,self.factors,on='jdate',how='inner')

            # Alpha_list:
            # res_list=[['Excess Return','1'],['CAMP alpha','1 + Mkt_RF'],['FF3 alpha','1 + Mkt_RF + SMB + HML'],\
            #         ['FFC alpha','1 + Mkt_RF + SMB + HML + MOM'],['FF5 alpha','1 + Mkt_RF + SMB + HML + RMW + CMA']]
            # res_type=['Expected Average Excess Return','CAMP','FF3','FFC','FF5']
            res_list=[['Excess Return','1'],['CAMP alpha','1 + Mkt_RF'],['FFC alpha','1 + Mkt_RF + SMB + HML + MOM'],['FF5 alpha','1 + Mkt_RF + SMB + HML + RMW + CMA']]
            res_type=['Expected Average Excess Return','CAMP','FFC','FF5']

            res_main=[]

            # Regression and Store results
            for i in y_list:
                for j0,j in enumerate(res_list):
                    # Time Series Regression
                    res=smf.ols(i + ' ~ ' + j[1],data=ret_future1_fac).fit(cov_type='HAC',cov_kwds={'maxlags':20})
                    res_main.extend([[i,j[0],res.params[0],res.bse[0],res.tvalues[0],res.pvalues[0]]])

            table2=pd.DataFrame(res_main,columns=['Y','Form','coef','std','tvalue','pvalue'])
            table2['Kind']='H-L: '+table2['Y'].str[-4:-2]
            table2['Group']=table2['Y'].str[-1:]
            table2['Var']=second_sort_list[g][1]
            
            # Set order
            order = pd.CategoricalDtype(second_sort_q_l[g][1],ordered=True)
            order1 = pd.CategoricalDtype([k[0] for k in res_list],ordered=True)
            table2['Group']=table2['Group'].astype(order)
            table2['Form']=table2['Form'].astype(order1)

            Table2_coef=table2.pivot_table(index=['Var','Group'],columns=['Kind','Form'],values=['coef'])
            Table2_p=table2.pivot_table(index=['Var','Group'],columns=['Kind','Form'],values=['pvalue'])
            Table2_std=table2.pivot_table(index=['Var','Group'],columns=['Kind','Form'],values=['std'])

            # Set pvalue stars ***
            Table2_p=pvalue_func(Table2_p)

            Table2_coef=Table2_coef.T.reset_index(level=[0],drop=True).T
            Table2_p=Table2_p.T.reset_index(level=[0],drop=True).T
            Table2_std=Table2_std.T.reset_index(level=[0],drop=True).T

            # Combine statistics
            for e in range(Table2_coef.shape[0]):
                l=(Table2_coef.iloc[e].round(3).astype(str)+''+Table2_p.iloc[e].astype(str)+'('+Table2_std.iloc[e].round(3).astype(str)+')').to_frame()
                
                if e==0:
                    l2=l.copy()
                else:
                    # print(l,l1)
                    l2=l2.join(l)

            # Combine all the table
            if g==0:
                Table2_final=l2.T
            else:
                Table2_final=pd.concat([Table2_final,l2.T])
            
            multi_index1=[('Market value', j) if i == 'me' else (i,j) for i, j in Table2_final.index ]
            multi_index2=[('Book-to-market', j) if i == 'beme' else (i,j) for i, j in multi_index1]
            Table2_final.index=pd.MultiIndex.from_tuples(multi_index2)


        ### Summary Statistics
        # Statistics
        ccm_monthly['Firm Number']=1
        ccm_monthly[sort_var[0]]=ccm_monthly[sort_var[0]].astype(pd.CategoricalDtype(sort_q_l[1],ordered=True))

        # Firm number: Average jdate (Time) firm coount.
        for g in range(len(second_sort_list)):

            firm_num=ccm_monthly.groupby([sort_var[0],second_var[g],'jdate'])[['Firm Number']].sum().reset_index()\
                [[sort_var[0],second_var[g],'Firm Number']]
            t1=pd.pivot_table(firm_num,index=[sort_var[0]], columns=[second_var[g]], values=['Firm Number'], aggfunc=np.mean)
            t1.columns=pd.MultiIndex.from_tuples([tuple([second_var[g]])+tuple([i[1]]) for i in t1.columns])
            t1.columns.names=['Averaged Firm Number','']

            if g==0:
                table_firm_num=t1.copy()
            else:
                table_firm_num=pd.concat([table_firm_num,t1],axis=1)


        notice='Notice: coef(p-value)(std): *p<.1; **p<.05; ***p<.01'

        return Table2_final,table_firm_num,notice

    # Dependent or Conditional Bivariate Sorting
    # Equal-weighted & Value-weighted portfolios high-low average excess return and FF5 alpha.
    def Table2_Dependent(self,future_return=1,sample_window=[1963,2023],\
               sort_var=['XXXport','VAR'],q_t='Quantile',q_df='NYSE',sort_q_l=[[0,0.3,0.7,1],['L','M','H']],maxlags=12,\
               second_sort_list=[['XXX1port','VAR'],['XXX2port','VAR'],['XXX3port','VAR']],\
               second_sort_q_t_df=[['Quantile','Self'],['Quantile','Self'],['Quantile','Self']],\
               second_sort_q_l=[[[0,0.5,1],['L','H']],[[0,0.5,1],['L','H']],[[0,0.5,1],['L','H']]]):
        # If we sort one variable from financial variables, we should remove it to avoid duplicated calculations later.
        self.financial_var1 = list(filter(lambda x: x != sort_var[1], self.financial_var))
        
        ccm_month_win=self.ccm_month_win.copy()
        # Time selection
        ccm_month_win=ccm_month_win[(ccm_month_win['year']>=sample_window[0])&(ccm_month_win['year']<=sample_window[1])]

        # Dropna of sort_var
        ccm_month_win.dropna(subset=[sort_var[1]],inplace=True)

        # Double Sorting 
        second_var=[]
        second_post=[]
        for g in range(len(second_sort_list)):
            try:
                # First Sorting
                ccm_month_win=AP_table.Sorting(self,ccm_month_win,sort_variable=second_sort_list[g],\
                                            q_tp=second_sort_q_t_df[g][0],q_data=second_sort_q_t_df[g][1],q_point=second_sort_q_l[g][0],q_label=second_sort_q_l[g][1])
                second_var.append(second_sort_list[g][0])

                # Dependent Sorting
                sort_var_b=[second_sort_list[g][1]+'_'+sort_var[0],sort_var[1]]
                ccm_month_win=ccm_month_win.groupby([second_sort_list[g][0]],group_keys=False)\
                    .apply(lambda x : AP_table.Sorting(self,x,sort_variable=sort_var_b,q_tp=q_t,q_data=q_df,q_point=sort_q_l[0],q_label=sort_q_l[1]))
                second_post.append(second_sort_list[g][1]+'_'+sort_var[0])

            except:
                Table2_final=pd.DataFrame()


        # store portfolio assignment as of Given Month
        month_win=ccm_month_win[['PERMNO','date','jdate','beme']+self.financial_var1+second_post+second_var]
        o=month_win.copy()
        o['ffyear']=month_win['jdate'].dt.year

        # merge back with monthly records
        crsp_monthly = self.crsp_monthly[['date','PERMNO','SHRCD','EXCHCD','retadj','me','wt','cumretx','ffyear','jdate']]
        ccm_monthly=pd.merge(crsp_monthly, 
                o[['PERMNO','ffyear','beme']+self.financial_var1+second_post+second_var], how='left', on=['PERMNO','ffyear'])

        # keeping only records that meet the criteria
        ccm_monthly=ccm_monthly[(ccm_monthly['wt']>0) & 
                ((ccm_monthly['SHRCD']==10) | (ccm_monthly['SHRCD']==11))]

        ccm_monthly['year']=ccm_monthly['jdate'].dt.year.copy()

        # Set time range
        ccm_monthly=ccm_monthly[(pd.to_datetime(ccm_monthly.jdate)>=datetime.datetime.strptime(str(sample_window[0])+'0731','%Y%m%d'))&\
                  (pd.to_datetime(ccm_monthly.jdate)<=datetime.datetime.strptime(str(sample_window[1]+1)+'0630','%Y%m%d'))]
        
        # Bivariate
        for g in range(len(second_sort_list)):
            # Equal-weighted & Value-weighted return rate.

            vw_ret=ccm_monthly.groupby(['jdate',second_var[g],second_post[g]],group_keys=True).apply(wavg, 'retadj','wt').reset_index().rename(columns={0: 'vwret'})\
                .pivot(index=['jdate'], columns=[second_var[g],second_post[g]], values='vwret')

            ew_ret=ccm_monthly.groupby(['jdate',second_var[g],second_post[g]],group_keys=True)['retadj'].mean().reset_index().rename(columns={'retadj': 'ewret'})\
                .pivot(index=['jdate'], columns=[second_var[g],second_post[g]], values='ewret')
            
            # print(ew_ret,vw_ret)
            
            # H-L group
            # Generate H-L return time series
            for order_g,s_g in enumerate(second_sort_q_l[g][1]):
                h_l_vw1=(vw_ret[(s_g,'H')]-vw_ret[(s_g,'L')])
                h_l_ew1=(ew_ret[(s_g,'H')]-ew_ret[(s_g,'L')])
                h_l_vw1.name='H_L_vw_'+s_g
                h_l_ew1.name='H_L_ew_'+s_g

                if order_g==0:
                    h_l_vw=h_l_vw1.copy()
                    h_l_ew=h_l_ew1.copy()
                else:
                    h_l_vw=pd.concat([h_l_vw,h_l_vw1],axis=1)
                    h_l_ew=pd.concat([h_l_ew,h_l_ew1],axis=1)

            # Combine with the factors time series.
            time_data=pd.merge(h_l_vw,h_l_ew,on='jdate',how='inner')

            # Second Table about the alpha of L to H groups
            time_data2=pd.concat([vw_ret,ew_ret],axis=1)
            time_data2.columns=[s2+'_vw'+'_'+s1 for s1 in second_sort_q_l[g][1]for s2 in sort_q_l[1]] + [s2+'_ew'+'_'+s1 for s1 in second_sort_q_l[g][1]for s2 in sort_q_l[1]]

            # Calculate excess return
            RF_data=pd.merge(time_data.reset_index(),self.factors[['jdate','RF']],on='jdate',how='inner').set_index('jdate')['RF']
            time_data=time_data.sub(RF_data,axis=0)
            time_data2=time_data2.sub(RF_data,axis=0)

            # Future 1 month return rate
            ret_future1=time_data.reset_index().copy()
            ret_future1['jdate']=time_data.reset_index()['jdate'].shift(future_return)
            ret_future1.dropna(inplace=True)

            ret_future2=time_data2.reset_index().copy()
            ret_future2['jdate']=time_data2.reset_index()['jdate'].shift(future_return)
            ret_future2.dropna(inplace=True)

            # Regression variables
            y_list=ret_future1.columns.tolist()[1:]
            y_list2=ret_future2.columns.tolist()[1:]

            # Future 1 month return rate combine factors
            ret_future1_fac=pd.merge(ret_future1,self.factors,on='jdate',how='inner')

            ret_future2_fac=pd.merge(ret_future2,self.factors,on='jdate',how='inner')

            # Alpha_list:
            # res_list=[['Excess Return','1'],['CAMP alpha','1 + Mkt_RF'],['FF3 alpha','1 + Mkt_RF + SMB + HML'],\
            #         ['FFC alpha','1 + Mkt_RF + SMB + HML + MOM'],['FF5 alpha','1 + Mkt_RF + SMB + HML + RMW + CMA']]
            res_list=[['Excess Return','1'],['CAMP alpha','1 + Mkt_RF'],['FFC alpha','1 + Mkt_RF + SMB + HML + MOM'],['FF5 alpha','1 + Mkt_RF + SMB + HML + RMW + CMA']]

            res_main=[]
            res_main2=[]
            # Regression and Store results
            for i in y_list:
                for j in res_list:
                    # Time Series Regression
                    res=smf.ols(i + ' ~ ' + j[1],data=ret_future1_fac).fit(cov_type='HAC',cov_kwds={'maxlags':maxlags})
                    
                    res_main.extend([[i,j[0],res.params[0],res.bse[0],res.tvalues[0],res.pvalues[0]]])


            # Regression and Store results
            for i in y_list2:
                for j in res_list:
                    # Time Series Regression
                    res=smf.ols(i + ' ~ ' + j[1],data=ret_future2_fac).fit(cov_type='HAC',cov_kwds={'maxlags':maxlags})
                    
                    res_main2.extend([[i,j[0],res.params[0],res.bse[0],res.tvalues[0],res.pvalues[0]]])

            ############################### Making Table: H-L groups ###############################
            table2=pd.DataFrame(res_main,columns=['Y','Form','coef','std','tvalue','pvalue'])

            table2['Kind']='H-L: '+table2['Y'].apply(lambda x: x.split('_')[-2])
            table2['Group']=table2['Y'].apply(lambda x: x.split('_')[-1])
            table2['Var']=second_sort_list[g][1]

            # Set order
            order = pd.CategoricalDtype(second_sort_q_l[g][1],ordered=True)
            order1 = pd.CategoricalDtype([k[0] for k in res_list],ordered=True)
            table2['Group']=table2['Group'].astype(order)
            table2['Form']=table2['Form'].astype(order1)

            Table2_coef=table2.pivot_table(index=['Var','Group'],columns=['Kind','Form'],values=['coef'])
            Table2_p=table2.pivot_table(index=['Var','Group'],columns=['Kind','Form'],values=['pvalue'])
            Table2_std=table2.pivot_table(index=['Var','Group'],columns=['Kind','Form'],values=['std'])

            # Set pvalue stars ***
            Table2_p=pvalue_func(Table2_p)

            Table2_coef=Table2_coef.T.reset_index(level=[0],drop=True).T
            Table2_p=Table2_p.T.reset_index(level=[0],drop=True).T
            Table2_std=Table2_std.T.reset_index(level=[0],drop=True).T

            # Combine statistics
            for e in range(Table2_coef.shape[0]):
                l=(Table2_coef.iloc[e].round(3).astype(str)+''+Table2_p.iloc[e].astype(str)+'('+Table2_std.iloc[e].round(3).astype(str)+')').to_frame()
                
                if e==0:
                    l2=l.copy()
                else:
                    # print(l,l1)
                    l2=l2.join(l)

            # Combine all the table
            if g==0:
                Table2_final=l2.T
            else:
                Table2_final=pd.concat([Table2_final,l2.T])
            
            multi_index1=[('Market value', j) if i == 'me' else (i,j) for i, j in Table2_final.index ]
            multi_index2=[('Book-to-market', j) if i == 'beme' else (i,j) for i, j in multi_index1]
            Table2_final.index=pd.MultiIndex.from_tuples(multi_index2)

            ############################### Making Tables: L to H ew and vw groups ###############################
            table22=pd.DataFrame(res_main2,columns=['Y','Form','coef','std','tvalue','pvalue'])
            table22['Kind1']=table22['Y'].apply(lambda x: x.split('_')[0])
            table22['Kind']=table22['Y'].apply(lambda x: x.split('_')[0])+':'+table22['Y'].apply(lambda x: x.split('_')[-2])
            table22['Group']=table22['Y'].apply(lambda x: x.split('_')[-1])
            table22['Var']=second_sort_list[g][1]

            # Set order
            order = pd.CategoricalDtype(second_sort_q_l[g][1],ordered=True)
            order1 = pd.CategoricalDtype([k[0] for k in res_list],ordered=True)
            table22['Group']=table22['Group'].astype(order)
            table22['Form']=table22['Form'].astype(order1)

            table22['Kind1']=table22['Kind1'].astype(pd.CategoricalDtype(sort_q_l[1],ordered=True))


            Table22_coef=table22.pivot_table(index=['Var','Group'],columns=['Kind','Form'],values=['coef'])
            Table22_p=table22.pivot_table(index=['Var','Group'],columns=['Kind','Form'],values=['pvalue'])
            Table22_std=table22.pivot_table(index=['Var','Group'],columns=['Kind','Form'],values=['std'])

            # Set pvalue stars ***
            Table22_p=pvalue_func(Table22_p)

            Table22_coef=Table22_coef.T.reset_index(level=[0],drop=True).T
            Table22_p=Table22_p.T.reset_index(level=[0],drop=True).T
            Table22_std=Table22_std.T.reset_index(level=[0],drop=True).T

            # Combine statistics
            for e in range(Table22_coef.shape[0]):
                l=(Table22_coef.iloc[e].round(3).astype(str)+''+Table22_p.iloc[e].astype(str)+'('+Table22_std.iloc[e].round(3).astype(str)+')').to_frame()
                
                if e==0:
                    l22=l.copy()
                else:
                    # print(l,l1)
                    l22=l22.join(l)

            # Combine all the table
            if g==0:
                Table22_final=l22.T
            else:
                Table22_final=pd.concat([Table22_final,l22.T])
            
            multi_index12=[('Market value', j) if i == 'me' else (i,j) for i, j in Table22_final.index ]
            multi_index22=[('Book-to-market', j) if i == 'beme' else (i,j) for i, j in multi_index12]
            Table22_final.index=pd.MultiIndex.from_tuples(multi_index22)

        # Sort Table22's columns
        def sort_key(x):
            kind_order = {'L': 1, 'M': 2, 'H': 3}
            return kind_order[x]
        sort_index=sorted(Table22_final.columns, key=lambda x: sort_key(x[0][:1]))
        Table22_final=Table22_final[sort_index]
        
        ### Summary Statistics
        # Statistics
        ccm_monthly['Firm Number']=1
        for g in range(len(second_post)):
            ccm_monthly[second_post[g]]=ccm_monthly[second_post[g]].astype(pd.CategoricalDtype(sort_q_l[1],ordered=True))

        # Firm number: Average jdate (Time) firm coount.
        for g in range(len(second_sort_list)):

            firm_num=ccm_monthly.groupby([second_post[g],second_var[g],'jdate'])[['Firm Number']].sum().reset_index()\
                [[second_post[g],second_var[g],'Firm Number']]
            t1=pd.pivot_table(firm_num,index=[second_post[g]], columns=[second_var[g]], values=['Firm Number'], aggfunc=np.mean)
            t1.columns=pd.MultiIndex.from_tuples([tuple([second_var[g]])+tuple([i[1]]) for i in t1.columns])
            t1.columns.names=['Averaged Firm Number','']

            if g==0:
                table_firm_num=t1.copy()
            else:
                table_firm_num=pd.concat([table_firm_num,t1],axis=1)


        notice='Notice: coef(p-value)(std): *p<.1; **p<.05; ***p<.01'

        return Table2_final,Table22_final,table_firm_num,notice
    
    #####################################################
    # Load Fama French Factors Month and Year Frequency #
    #####################################################
    def F_F_Factors_MY(self,type='monthly',file_FF5='F_F/F-F_5_Factors_monthly.csv',file_MOM='F_F/F-F_Momentum_monthly.CSV'):
        # Fama-French 5 factors
        FF5=pd.read_csv(file_FF5)
        # Momentum factor
        MOM=pd.read_csv(file_MOM)
        if type=='monthly':
            FF5['jdate']=FF5['jdate'].apply(lambda x: datetime.datetime.strptime(str(x),'%Y%m'))+MonthEnd()
            MOM['jdate']=MOM['jdate'].apply(lambda x: datetime.datetime.strptime(str(x),'%Y%m'))+MonthEnd()
            MOM.columns=['jdate','MOM']

        elif type=='annually':
            FF5['jdate']=FF5['jdate'].apply(lambda x: datetime.datetime.strptime(str(x),'%Y'))
            MOM['jdate']=MOM['jdate'].apply(lambda x: datetime.datetime.strptime(str(x),'%Y'))
            MOM.columns=['jdate','MOM']
        
        factors=pd.merge(FF5,MOM,on='jdate',how='inner')
        if type=='monthly':
            factors=factors.rename(columns={'Mkt-RF':'Mkt_RF'})
        elif type=='annually':
            factors=factors.rename(columns={'Mkt-RF':'Mkt_RF','jdate':'year'})
            factors['year']=factors['year'].astype('int64')

        return factors

    ########################################
    #      Add Control Variables Block     #
    ########################################
    def control(self):

        # merge back with monthly records
        crsp_monthly = self.crsp_monthly[['date','PERMNO','SHRCD','EXCHCD','retadj','me','wt','cumretx','year','month','jdate','1+retx','ffyear']]
        factor_monthly=AP_table.F_F_Factors_MY(self,type='monthly',file_FF5='F_F/F-F_5_Factors_monthly.csv',file_MOM='F_F/F-F_Momentum_monthly.CSV')
        crsp_monthly=pd.merge(crsp_monthly,factor_monthly,on='jdate',how='inner')

        beta=crsp_monthly.groupby('PERMNO').apply(lambda df: (df['retadj']-df['RF']).rolling(window=60,min_periods=24).cov(df['Mkt_RF'])/df['Mkt_RF'].rolling(window=60,min_periods=24).var())
        crsp_monthly['Beta5Y_monthly']=beta.reset_index(level=0,drop=True)
        mom=crsp_monthly.groupby('PERMNO').apply(lambda df: df['retadj'].shift(1).rolling(window=11).sum())
        crsp_monthly['Momentum_monthly']=mom.reset_index(level=0,drop=True)

        # Stock Volatility: Notice ffyear groupby
        stock_volatility=crsp_monthly.groupby(['PERMNO','ffyear']).apply(lambda df: df['retadj'].std()).reset_index()
        stock_volatility['Stock_volatility']=stock_volatility.groupby(['PERMNO'],group_keys=False)[0].apply(lambda x: x.shift(1))
        stock_volatility=stock_volatility.drop(columns=[0])
        stock_volatility['year']=stock_volatility['ffyear']+1
        stock_volatility.drop(columns=['ffyear'],inplace=True)
        
        crsp_monthly.drop(columns=factor_monthly.columns[1:].tolist(),inplace=True)

        # Calculate Annual return
        crsp_monthly['1+retx_shift_portfolio']=crsp_monthly.groupby(['PERMNO'])['1+retx'].shift(self.portfolio_month)

        crsp_monthly['Realized_ret_annual']=(crsp_monthly.groupby(['PERMNO','ffyear'])['1+retx_shift_portfolio'].cumprod()-1)*100

        # Calculate Annual return
        crsp_monthly['ret_annual']=(crsp_monthly.groupby(['PERMNO','ffyear'])['1+retx'].cumprod()-1)*100

        # Portfolio month = 6, which means ffyear is last year. (Data: ffyear, Match Identify: year "ffyear+1")
        # The year in jp data is fyear+1.
        crsp_annually=crsp_monthly[crsp_monthly['month']==self.portfolio_month]

        ccm_new=pd.merge(self.ccm_month_win[['PERMNO','gvkey','year','beme','sic']+self.financial_var], crsp_annually, how='left',\
                        left_on=['PERMNO','year'],right_on=['PERMNO','year'])

        ccm_new=pd.merge(ccm_new,stock_volatility,on=['PERMNO','year'],how='left')

        self.financial_var2=self.financial_var+['Beta5Y_monthly','Momentum_monthly','Stock_volatility']
        self.ccm_new=ccm_new
        
        return self.ccm_new

    #################################
    ### Load Industry Information ###
    #################################
    def Industry(self,industry_file='Siccodes49.txt'):
        ### Industry Information ###
        data = []
        file = open(industry_file,'r',encoding='UTF-8')
        file_data = file.readlines() 
        for row in file_data:
            data.append(row)

        siccode=pd.DataFrame(data)[0].str.strip().str.split(' ',1,expand=True)
        siccode=pd.concat([siccode[0].str.split('-',expand=True),siccode[1]],axis=1)
        siccode=siccode[siccode[0]!='']
        siccode.columns=[0,1,2]

        siccode['Group']=0
        siccode['Group']=siccode['Group'].mask(siccode[1].isna(),siccode[0])
        x=siccode.Group.tolist()
        for i in range(len(x)):
            if x[i]==0:
                x[i]=x[i-1]

        siccode['Group']=x

        siccode['Interval']=np.nan

        siccode.loc[siccode[1].notna(),'Interval']=siccode[siccode[1].notna()].apply(lambda x: (int(x[0]),int(x[1])),axis=1)

        siccode.columns=['Min','Max','Name','Group','Interval']
        siccode.astype({'Min':'Int64','Max':'Int64'})

        sicd=siccode.dropna().Interval.tolist()
        siclabel=siccode.dropna().Group.tolist()

        sicinterval=pd.IntervalIndex.from_tuples(sicd,closed='both')

        sicdint=siccode[siccode['Max'].notna()].apply(lambda x: [int(x['Min']),int(x['Max'])],axis=1).tolist()

        dictsic={}
        for i in range(len(sicdint)):
            dictsic[str(sicdint[i])]=siclabel[i]
        dictsic
        
        self.ccm_new['industry']=pd.Series(pd.Categorical(self.ccm_new.sic,sicinterval)).astype('str').map(dictsic)

        self.ccm_new.industry=self.ccm_new.industry.astype('Int64')

        self.ccm_new[self.ccm_new['industry'].isna()][['sic','industry']]

        ### Exclude nan
        ccm_new_industry=self.ccm_new[self.ccm_new['industry'].notna()]
        
        # Add industry name
        industry_list=siccode[siccode['Interval'].isna()][['Name','Group']].reset_index(drop=True)
        industry_list.columns=['industry_name','industry']
        industry_list['industry']=industry_list['industry'].astype('int64')
        industry_list

        self.ccm_new_industry2=pd.merge(ccm_new_industry,industry_list,on=['industry'])

        return self.ccm_new_industry2
    
    # Fama-Macbeth Regression
    def Table3(self,y_lag=[1,2,3,6,9,12],Industry=None):
        '''
        A function for generating lag return data for Fama-MacBeth regression.

        Parameters
        ----------
        y_lag: list
            A list of lag months return that we want to add as y variable in regression.
            variables, e.g., [1,2,3,6,9,12].
        Industry: None, True
            Whether add industry information? Because some data may without the industry's category and would be drop in this situation.
        '''
        
        if Industry is not None:
            self.ccm_new=self.Industry(industry_file='Siccodes49.txt')
            industry_var=['industry','industry_name']
        else:
            industry_var=[]
            pass

        # Use monthly data calculating the 
        ccm_month_win=self.ccm_new.copy()
        # ccm_month_win=ccm_month_win[(ccm_month_win['year']>=sample_window[0])&(ccm_month_win['year']<=sample_window[1])]

        # store portfolio assignment as of Given Month
        # month_win=ccm_month_win[['PERMNO','date','jdate','beme']+self.financial_var2+x]
        o=ccm_month_win.copy()
        o['year']=o['jdate'].dt.year

        # merge back with monthly records
        crsp_monthly = self.crsp_monthly[['date','PERMNO','SHRCD','EXCHCD','retadj','me','wt','cumretx','ffyear','jdate']]

        crsp_monthly_fac=pd.merge(crsp_monthly,self.factors,on='jdate',how='inner')

        # if 'Beta' in control:
        beta=crsp_monthly_fac.groupby('PERMNO').apply(lambda df: (df['retadj']-df['RF']).rolling(window=60,min_periods=24).cov(df['Mkt_RF'])/df['Mkt_RF'].rolling(window=60,min_periods=24).var())
        crsp_monthly_fac['Beta']=beta.reset_index(level=0,drop=True)

        # if 'Momentum' in control:
        mom=crsp_monthly_fac.groupby('PERMNO').apply(lambda df: df['retadj'].shift(1).rolling(window=11).sum())
        crsp_monthly_fac['Momentum']=mom.reset_index(level=0,drop=True)

        crsp_monthly_fac['year']=pd.to_datetime(crsp_monthly_fac['jdate']).dt.year

        ccm_monthly=pd.merge(crsp_monthly_fac, 
            o[['PERMNO','year','beme']+self.financial_var2+industry_var], how='left', on=['PERMNO','year'])
        
        ccm_monthly['me']=ccm_monthly['me']/1000000  # dollars to million dollars
        
        # Calculate excess return
        ccm_monthly['exretadj']=ccm_monthly['retadj']-ccm_monthly['RF']

        # Calculate the ret_lag
        for y in y_lag:
            ret=ccm_monthly.groupby(['PERMNO'],group_keys=True)['exretadj'].apply(lambda x: x.shift(-y))
            ccm_monthly['exret_'+str(y)]=ret.reset_index(level=0,drop=True)

        self.y_lag=y_lag

        self.ccm_table3=ccm_monthly

        # c=''
        # for j in control:
        #     c=c+' + '+j
        # regress_list=[['exret_'+str(i) + ' ~ 1 + ' + x[0] + c] for i in y_lag]

        # for r in regress_list:
        #     mod = FamaMacBeth.from_formula(r[0], data=ccm_monthly)
        #     res = mod.fit(cov_type= 'kernel',debiased = False, bandwidth = maxlags)
        #     print(res.summary)
        return self.ccm_table3

    # Fama-MacBeth Regression
    def FamaMacBeth(self,time_series=['19630101','20231231'],x=['CitMaxApPurEffBrownEP'],control=['me','beme','Beta','Momentum'],maxlags=None,FE=None):
        # Set time range
        ccm_table3_2=self.ccm_table3[(pd.to_datetime(self.ccm_table3.jdate)>=datetime.datetime.strptime(time_series[0],'%Y%m%d'))&\
                  (pd.to_datetime(self.ccm_table3.jdate)<=datetime.datetime.strptime(time_series[1],'%Y%m%d'))]
        
        # Fixed Effect
        fe_varlist=[]

        if FE is None:
            pass
        else:
            self.fe_effect=True
            for fe in FE:
                # Generate dummy variables and Avoid Perfect Parallel
                fe_dummy=pd.get_dummies(ccm_table3_2[fe],drop_first=True,dtype=float,dummy_na=False)
                fe_varlist.extend(fe_dummy.columns.tolist())

                # Combine
                ccm_table3_2=pd.concat([ccm_table3_2,fe_dummy],axis=1)

        
        ccm_table3_2 = ccm_table3_2.set_index(['PERMNO', 'jdate']) # multi-index

        ccm_table3_2[x+['exret_'+str(i) for i in self.y_lag]]=ccm_table3_2[x+['exret_'+str(i) for i in self.y_lag]].astype(float)

        Table3=FamaMacBeth_summary(ccm_table3_2,reg_lst=[['exret_'+str(self.y_lag[0])]+x+fe_varlist]+[['exret_'+str(i)]+x+control+fe_varlist for i in self.y_lag],\
                                   reg_order=x+control+fe_varlist,\
                                   reg_names=['exret_t+'+str(self.y_lag[0])]+['exret_t+'+str(i) for i in self.y_lag],\
                                    lags=maxlags)
        
        return Table3

def assign_exchange(exchcd):
    if exchcd in [1, 31]:
        return "NYSE"
    elif exchcd in [2, 32]:
        return "AMEX"
    elif exchcd in [3, 33]:
        return "NASDAQ"
    else:
        return "Other"

def convert_currency(var):
    try:
        var = float(var)
    except:
        var = var
    return var

# Calculate one-sorting portfolio based on q_data like "NYSE", and set quantiles and its labels.
def qfunc(x,q_data,sort_col,q,label,q_type='Quantile'):
    if isinstance(q, int):
        q=np.linspace(0,1,q+1)

    index1=x.index.copy()
    
    # IF "NYSE" and not datas at x.name time
    try: 
        df=q_data.groupby(['jdate'])[sort_col].get_group(x.name)
    except:
        print('No datas at time ', x.name)
        x.iloc[:]=np.nan
        return x
    
    # q_type: "Quantile" or "Censore"

    if q_type=='Quantile':
        try:
            f=pd.Categorical(x,(pd.qcut(df,q).dtypes).categories.tolist())
        except:
            q_df=[df.quantile(q[i]) for i in range(0,len(q),1)]
            # Justify whether consecutive three numbers are the same
            c3=[ True if (q_df[i]==q_df[i+1]) and (q_df[i+1]==q_df[i+2]) else False for i in range(0,len(q)-2,1)]
            # If exist three numbers are the same, it's unable to distinguish groups. Thus, it should be sent to error time.
            if any(c3):
                print('Error time with three consecutive quantiles',q_df,c3)
                # If will return error and breakup to the next part
                bins=pd.IntervalIndex.from_tuples([(q_df[i],q_df[i+1]) for i in range(0,len(q)-1,1) ])
                f=pd.Categorical(x,bins)
            else:
                # Justify whether consecutive two numbers are the same
                c2=[ True if (q_df[i]==q_df[i+1]) else False for i in range(0,len(q)-1,1)]
                # If the situation is happened in the middle, it should be errored.
                if any(c2[1:]):
                    print('Error time with two consecutive quantiles:',x.name,)
                    bins=pd.IntervalIndex.from_tuples([(q_df[i],q_df[i+1]) for i in range(0,len(q)-1,1) ])
                    f=pd.Categorical(x,bins)
                # If happened at the first and second quantile, we modify the interval.
                elif c2[0]:
                    print('Error time with the first and second consecutive quantiles:',x.name,)
                    bins=pd.IntervalIndex.from_tuples([(q_df[i]-0.01,q_df[i+1]) if i==0 else (q_df[i],q_df[i+1]) for i in range(0,len(q)-1,1) ])
                    f=pd.Categorical(x,bins)
                else:
                    bins=pd.IntervalIndex.from_tuples([(q_df[i],q_df[i+1]) for i in range(0,len(q)-1,1) ])
                    f=pd.Categorical(x,bins)

        f=pd.Series(f,index=index1)
        f=f.cat.rename_categories(label)
    
    # If Censored, we use the first quantile as a group, then excluding these data and dividing groups.
    elif q_type=='Censored':
        # Generate the interval
        d=[]
        for i in range(0,len(q)-1,1):
            g1=df.quantile(q[0])

            if i==0:
                d.append(pd.Interval(g1-0.001,g1,closed='right'))
            elif i==1:
                d.append(pd.Interval(g1,df[df!=g1].quantile(q[i+1]),closed='right'))
            else:
                d.append(pd.Interval(df[df!=g1].quantile(q[i]),df[df!=g1].quantile(q[i+1]),closed='right'))
        
        f=pd.Categorical(x,d)
        f=pd.Series(f,index=index1)
        f=f.cat.rename_categories(label)

    return f

# function to calculate value weighted return
def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan

def pvalue_func(y):
    x=y.to_numpy()
    x[x<0.01]=1
    x[x<0.05]=2
    x[x<0.1]=3
    x[(x<1)&(x>=0.1)]=4

    y=y.replace(1,'(***)')
    y=y.replace(2,'(**)')
    y=y.replace(3,'(*)')
    y=y.replace(4,'()')

    return y

def FamaMacBeth_summary(DF,
                        reg_lst,
                        reg_order,
                        reg_names=None,
                        params_format='{:.3f}',
                        tvalues_format='{:.2f}',
                        lags=None):

    '''
    A function for Fama-MacBeth regression and results summary.

    Parameters
    ----------
    DF: DataFrame
        A panel date of which multi-index is stock and month (datetime64[ns]),
        containing all the dependent and independent variables.
    reg_lst: list
        A list containing multiple lists of dependent variable and independent
        variables, e.g., [['Y', 'X1', ...],..., ['Y', 'X1', ...,]].
    reg_order: list
        The order of independent variables in result table.
    reg_names: list
        The names for each regression.
    params_format: str
        The number of decimal places for parameters, e.g., '{:.3f}'.
    tvalues_format: str
        The number of decimal places for t-values, e.g., '{:.2f}'.
    lags: number
        If none, would use the calculation
    '''

    # Create a DataFrame
    rows = sum([[var, f'{var}_t'] for var in ['const'] + reg_order], [])

    reg_name1 = [f'({i+1})' for i in range(len(reg_lst))]

    show = pd.DataFrame(index=rows, columns=reg_name1)

    for reg, reg_name in zip(reg_lst, reg_name1):
        df = DF.loc[:, reg].copy().dropna()
        T = len(df.index.get_level_values(df.index.names[1]).unique())
        if lags is None:
            lag = math.floor(4*(T/100)**(2/9))
        else:
            lag = lags
        
        fmb = FamaMacBeth(df[reg[0]], sm.add_constant(df[reg[1:]]))
        # Newey-West adjust
        fmb = fmb.fit(cov_type='kernel', bandwidth=lag)

        # params, tvalues(tstats) and pvalues
        params = fmb.params
        tvalues = fmb.tstats
        pvalues = fmb.pvalues

        # Obs.
        total_obs = fmb.nobs
        # mean_obs = fmb.time_info['mean']

        # average rsquared_adj
        dft = df.reset_index(level=df.index.names[0], drop=True).copy()
        rsquared_adj = []
        for month in dft.index.unique():
            dftm = dft.loc[month].copy()
            ols = sm.OLS(dftm[reg[0]], sm.add_constant(dftm[reg[1:]])).fit()
            rsquared_adj.append(ols.rsquared_adj)
        ar2a = np.mean(rsquared_adj)

        # params and significance
        ps_lst = []
        for param, pvalue in zip(params, pvalues):
            param = params_format.format(param)
            if (pvalue <= 0.1) & (pvalue > 0.05):
                param = param + '*'
            elif (pvalue <= 0.05) & (pvalue > 0.01):
                param = param + '**'
            elif pvalue <= 0.01:
                param = param + '***'
            ps_lst.append(param)

        # params and tvalues
        tvalues = [tvalues_format.format(t) for t in tvalues]
        t_lst = [f'({t})' for t in tvalues]
        pt_lst = [[i, j] for i, j in zip(ps_lst, t_lst)]

        # put them in place
        for var, pt in zip(['const'] + reg[1:], pt_lst):
            show.loc[var, reg_name] = pt[0]
            show.loc[f'{var}_t', reg_name] = pt[1]
        show.loc['No. Obs.', reg_name] = str(total_obs)
        show.loc['Adj. R²', reg_name] = '{:.2f}%'.format(ar2a * 100)

    rename_index = sum([[var, ''] for var in ['Intercept'] + reg_order], [])
    rename_index = ['Market value' if  i == 'me' else i for i in rename_index]
    rename_index = ['Book-to-market' if i == 'beme' else i for i in rename_index]
    
    show.index = rename_index + ['No. Obs.', 'Adj. R²']
    
    if reg_names is not None:
        col=pd.MultiIndex.from_tuples([(reg_name1[i],reg_names[i]) for i in range(len(reg_name1))])
        show.columns=col
        
    show=show.dropna(axis=0, how='all').fillna('')
    show=pd.concat([show.iloc[2:-2,:],show.iloc[:2,:],show.iloc[-2:,:]])

    return show
