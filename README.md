# Rossmann Store Sales Forecasting

**Overview**

This project focuses on building a machine learning model to forecast sales for Rossmann Pharmaceuticals stores six weeks in advance. This will empower the finance team with valuable insights to make informed business decisions.

**Business Need**

Rossmann Pharmaceuticals relies on manual sales forecasting by store managers, which can be subjective and inconsistent. The finance team requires a more accurate and reliable method for predicting sales across all stores.

**Data and Features**

* **Source:** Rossmann Store Sales | Kaggle 
* **Key Features:**
    * `Sales`: Target variable (daily turnover).
    * `Customers`: Number of customers on a given day.
    * `Open`: Store open status (0: closed, 1: open).
    * `StateHoliday`: Indicates state holidays (a: public, b: Easter, c: Christmas, 0: None).
    * `SchoolHoliday`: Indicates school closures.
    * `StoreType`: Store model (a, b, c, d).
    * `Assortment`: Assortment level (a: basic, b: extra, c: extended).
    * `CompetitionDistance`: Distance to the nearest competitor.
    * `CompetitionOpenSince[Month/Year]`: Opening date of the nearest competitor.
    * `Promo`: Indicates whether a store is running a promo.
    * `Promo2`: Indicates participation in a continuous promotion.
    * `Promo2Since[Year/Week]`: Start date of Promo2.
    * `PromoInterval`: Describes the intervals when Promo2 is started.

**Project Goals**

* Develop a robust machine learning model to accurately predict sales six weeks in advance.
* Build an end-to-end product that delivers these predictions to finance analysts.
* Improve decision-making and resource allocation within the finance team.
