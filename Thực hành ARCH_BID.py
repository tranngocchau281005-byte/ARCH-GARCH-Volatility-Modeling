# Bước 1: Import thư viện 
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Bước 2: Xử lý dữ liệu đầu vào
t1 = 'C:/Users/Chau/OneDrive - THPT Nguyễn Công Trứ/Attachments/TC định lượng/4_Trần Ngọc Châu_BID.csv'
BID = pd.read_csv(t1,sep=";")
print(BID)
print(BID.info())
# Chuyển time thành datetime
BID['time'] = pd.to_datetime(BID['time'])
print(BID.info())
# Sắp xếp để đảm bảo thứ tự thời gian
BID = BID.sort_values(by='time') 
# Kiểm tra các giá trị trùng lặp 
print("Hàng trùng lặp:") 
print(BID[BID.duplicated()]) 
print("=================================") 
print("Số lượng hàng trùng lặp:", BID.duplicated().sum()) 
# Xóa các hàng trùng lặp 
BID = BID.drop_duplicates() 
print("DataFrame sau khi loại bỏ trùng lặp:") 
print(BID) 
print("=================================") 
print(BID.info())
# Vẽ đồ thị giá đóng cửa
BID['close'].plot()
plt.show()

# Bước 3: Tính tỷ suất sinh lợi hằng ngày và kiểm tra giá trị lạ
returns = BID['close'].pct_change()
returns 
# Xóa NaN
returns = returns.dropna()
returns 
# Kiểm tra lại các giá trị lạ 
print("Có NaN không? ", returns.isna().any().any())
print("Có giá trị 0 không? ", (returns == 0).any().any())
print("Có giá trị inf hoặc -inf không? ", np.isinf(returns.values).any())
# Vẽ đồ thị lợi nhuận theo ngày
returns.plot(figsize=(10,5))
plt.show()

# Bước 4: Kiểm định hiệu ứng ARCH
!pip install arch
from arch import arch_model
# 1. Kiểm tra hiệu ứng ARCH:
model1 = arch_model(returns, mean='Zero', vol='ARCH', p=1, o=0, q=0)  

# 6. Estimate the model and print the summary:
model1_fitted = model1.fit(disp='off')
print(model1_fitted.summary())
#alpha[1]-> xet P>|t| -> -0.000000... -> có ý nghĩa thống kê, có hiêu ứng Arch

# 7. Plot the residuals and the conditional volatility:
model1_fitted.plot(annualize='D')
plt.show()

# Kết quả
"""**TIẾP TỤC VỚI MÔ HÌNH GARCH**"""

# 1. Specify the GARCH model:
model2 = arch_model(returns, mean='Zero', vol='GARCH', p=1, o=0, q=1)

# 2. Estimate the model and print the summary:
model2_fitted = model2.fit(disp='off')
print(model2_fitted.summary())
# P>|t| co 0 or e- la 10 mu tru nghia la gia tri rat nho co y nghia thong ke

# Hiển thị đồ thị mới
model2_fitted.plot(annualize='D')
plt.show()
