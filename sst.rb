require 'numo/narray'
require 'numo/linalg/autoloader'
require "numo/linalg/linalg"
require 'csv'

csvdata=CSV.table('abnormal_frequency.csv')
data = csvdata[:data]

def extract_matrix(data, start_data, end_data, w)
    row = w
    column = end_data - start_data + 1
    # 後で転置するために渡す引数の受け渡しを逆にする。
    # 演算結果を一致させる為に行った処理だが、演算として適切か？
    matrix = Numo::DFloat.new(column,row)
    i = 0
    matrix_array = Array.new
    for t in start_data-1..end_data-1
        matrix_array.push(data[t..t-1+row])
        matrix[]= matrix_array
        i += 1
    end
return matrix.transpose
end

# 入力されたデータがNarrayデータ形式でないなら変換
data ||= Numo::NArray[data]

# 計算に使う変数を初期化

# 異常検知のウィンドウサイズを50に設定
w ||= 50
m ||= 2

k ||= w / 2
L ||= k / 2
T = data.length

start_cal = k + w
end_cal = T - L + 1

change_scores = Numo::DFloat.zeros(data.length)

for t in start_cal..end_cal  do
    start_tra = t - w - k + 1
    end_tra = t - w
    tra_matrix = extract_matrix(data, start_tra, end_tra, w)
    start_test = start_tra + L
    end_test = end_tra + L
    test_matrix = extract_matrix(data, start_test, end_test, w)
    _, u_tra, _  = Numo::Linalg.svd(tra_matrix)
    _, u_test, _  = Numo::Linalg.svd(test_matrix) 

    u_tra_m = u_tra[0..u_tra[true,0].length-1,0..m-1]
    u_test_m = u_test[0..u_test[true,0].length-1,0..m-1]
    
    # 訓練用データの転置行列と検証用データのDOT積を計算し、特異値分解を行う
    s = Numo::Linalg.svd(Numo::Linalg.matmul(u_tra_m.transpose,u_test_m))
    change_scores[t] = 1 - s[0][0]
end

for t in change_scores
     p t
end