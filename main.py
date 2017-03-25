#!/usr/bin/python
import sys
import numpy as np
import settings as s
import fun


def main():
    """
    メイン関数
    処理の羅列
    """
    # 引数チェック
    if len(sys.argv) != 2:
        print("Usage: $ python %s [user's number]" % sys.argv[0])
        quit()

    # 引数より、Active User を定義
    active_user_id = int(sys.argv[1])
    # データセットを行列に格納
    database = fun.input_dataset(s.FILE_NAME)

    # アクティブユーザと各ユーザのピアソン相関を計算
    # pearson_list に格納
    pearson_list = np.zeros(s.N_USERS)
    for user_id in range(s.N_USERS):
        # 自分自身との相関はとらない
        if user_id == active_user_id:
            pearson_list[user_id] = 0.0
        else:
            pearson_list[user_id] = fun.pearson(database[active_user_id], database[user_id])

    # アイテムの評価値を予測する
    # 予測した評価値は item_ratings に格納
    item_ratings = np.zeros(s.N_ITEMS)
    for item in range(s.N_ITEMS):
        # 本人が評価済みのアイテムの評価値は-10
        if database[active_user_id][item] != 0:
            item_ratings[item] = -10.0
        else:
            item_ratings[item] = fun.predict(database, pearson_list, active_user_id, item)

    # 推薦アイテムを表示する
    fun.print_recommend(item_ratings)


if __name__ == '__main__':
    main()
