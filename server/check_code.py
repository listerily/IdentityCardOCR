def check_id_length(n):
    if len(str(n)) != 18:
        print("只支持18位身份证号查询")
        return False
    else:
        return True


def check_id_data(n):
    birth_year = None
    birth_month = None
    birth_date = None

    var = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    var_id = ['1', '0', 'x', '9', '8', '7', '6', '5', '4', '3', '2']
    # n = str(n)
    sum = 0
    # if int(n[16]) % 2 == 0:
    #     gender = "女"
    #     same = int(int(n[16]) / 2)
    # else:
    #     gender = "男"
    #     same = int((int(n[16]) + 1) / 2)
    for i in range(0, 17):
        sum += int(n[i]) * var[i]
    sum %= 11
    if (var_id[sum]) == str(n[17]):
        birth_year = n[6:10]
        birth_month = n[10:12]
        birth_date = n[12:14]
        print("身份证号规则核验通过，校验码是：", var_id[sum])
        print("出生于：", n[6:10], "年", n[10:12], "月", n[12:14], "日")
        return {
            'year': birth_year,
            'month': birth_month,
            'date': birth_date,
        }
    else:
        return None


def check_id_code(n):
    if check_id_length(n):
        results = check_id_data(n)
        if results is not None:
            return results
        else:
            return None
    else:
        return None


if __name__ == '__main__':
    code = input("请输入18位身份证号:")
    print(check_id_code(code))
