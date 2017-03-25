import csv
import numpy as np
import settings as s


def input_dataset(file_name):
    """
    データセットを行列に格納する
    user_id, item_id, rating, timestamp
    の4つのカラムから成るファイルを読み込む

    :param str file_name: userがitemを評価したratingが入ったファイルの場所
    :rtype: numpy.ndarray
    :return: 行がuser_id、列がitem_id、値がratingの行列
    """

    data = np.zeros((s.N_USERS, s.N_ITEMS))
    # temp: 一時読み込み用リスト
    temp = []

    # ファイルを読み込んでtempに入れる
    with open(file_name, 'r', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            temp.append(row)

    # tempのデータを所定の形式に変換
    for i in temp:
        data[int(i[0])-1][int(i[1])-1] = int(i[2])

    return data


def get_item_set(u_ratings):
    """
    user が評価したアイテムの集合を得る

    :param numpy.ndarray u_ratings: userのアイテム評価値の配列
    :rtype: set
    :return: user が評価したアイテムの集合
    """
    items = set()
    for (n, value) in enumerate(u_ratings):
        if value != 0:
            items.add(n)

    return items


def avg_item_rating(rating_array, item_set):
    """
    集合に含まれるアイテムの評価値の平均を計算する
    :param numpy.ndarray rating_array: ユーザの評価値を格納した配列
    :param set item_set: 評価の平均を取るアイテムの集合
    :rtype: float
    :return: 集合item_setに含まれているアイテムの評価値の平均
    """
    summation = 0.0
    for item in item_set:
        summation += float(rating_array[item])

    try:
        out = summation / float(len(item_set))
    except ZeroDivisionError as e:
        print(type(e))
    else:
        return out


def pearson(array_u1, array_u2):
    """
    ピアソン相関を計算する

    :param numpy.ndarray array_u1: userのアイテム評価値の配列1
    :param numpy.ndarray array_u2: userのアイテム評価値の配列2
    :rtype: float
    :return: Pearson 相関の値
    """
    # 返り値の定義
    out = 0.0

    # 双方とも共通に評価しているアイテムの集合をとる
    inter_sec = get_item_set(array_u1).intersection(get_item_set(array_u2))

    # 共通のアイテムが1以下ならば、相関は0
    if len(inter_sec) <= 1:
        return out

    # 実際に計算するための一時保存変数
    temp1 = temp2 = temp3 = 0.0
    # 共通に評価しているアイテムの評価平均
    avg_1 = avg_item_rating(array_u1, inter_sec)
    avg_2 = avg_item_rating(array_u2, inter_sec)
    # 総和計算のためのループ
    for i in inter_sec:
        temp1 += (float(array_u1[i]) - avg_1) ** 2
        temp2 += (float(array_u2[i]) - avg_2) ** 2
        temp3 += (float(array_u1[i]) - avg_1) * (float(array_u2[i]) - avg_2)

    temp4 = np.sqrt(temp1) * np.sqrt(temp2)
    if temp4 != 0:
        out = (temp3 / temp4)
    else:
        out = 0.0

    return out


def predict(database, p_list, a_user_id, item_id):
    """
    アイテムの評価値を予測する

    :param numpy.ndarray database: データセット
    :param numpy.ndarray p_list: Active Userと他のユーザのピアソン相関リスト
    :param int a_user_id: アクティブユーザのID
    :param int item_id: 評価値を予測したいアイテムID
    :rtype: float
    :return: 未評価アイテムの予測評価値
    """
    # Active User の全評価アイテムに対する平均評価値を求める
    a_avg_of_all = avg_item_rating(database[a_user_id], get_item_set(database[a_user_id]))

    # item を評価したユーザの集合を得る
    usr_set = set()
    for (index, u) in enumerate(range(s.N_USERS)):
        if database[u][item_id] != 0:
            usr_set.add(index)

    temp1 = 0.0
    temp2 = 0.0
    for u in usr_set:
        # 相関が0ならば加算不要
        if p_list[u] == 0.0:
            continue

        # 双方とも共通に評価しているアイテムの集合をとる
        inter_sec = get_item_set(database[a_user_id]).intersection(get_item_set(database[u]))
        temp1 += p_list[u] * (database[u][item_id] - avg_item_rating(database[u], inter_sec))
        temp2 += np.fabs(p_list[u])

    if temp2 != 0:
        out = a_avg_of_all + (temp1 / temp2)
    else:
        out = 0.0

    return out


def print_recommend(ratings):
    """
    評価値が最も高いアイテムを表示する

    :param numpy.ndarray ratings: 予測評価値が格納された配列
    :return:
    """
    maximum = 1.0
    item_num = 0

    for (num, a) in enumerate(ratings):
        if maximum < a:
            maximum = a
            item_num = num

    print('おすすめの映画は', item_num, ', 予測評価値は', maximum, 'です。')
