def check_id_length(n):
    if len(str(n)) != 18:
        return False
    else:
        return True


def checkX(n):
    for i in range(0, 16):
        if n[i] == 'X':
            return False
    return True


def year_is_legal(year):
    if 1900 < int(year) < 2100:
        return True
    else:
        return False


def is_ordinary_year(year):
    if int(year) % 4 == 0 and int(year) % 100 != 0 or int(year) % 400 == 0:
        return False
    else:
        return True


def month_is_legal(month):
    if 0 < int(month) < 13:
        return True
    else:
        return False


def days(year, month):
    if (is_ordinary_year(year)):
        days = {
            '01': 31,
            '02': 28,
            '03': 31,
            '04': 30,
            '05': 31,
            '06': 30,
            '07': 31,
            '08': 31,
            '09': 30,
            '10': 31,
            '11': 30,
            '12': 31
        }
    else:
        days = {
            '01': 31,
            '02': 29,
            '03': 31,
            '04': 30,
            '05': 31,
            '06': 30,
            '07': 31,
            '08': 31,
            '09': 30,
            '10': 31,
            '11': 30,
            '12': 31
        }
    return days.get(month)


def date_is_legal(year, month, date):
    if 0 < int(date) < days(year, month):
        return True
    else:
        return False


def check_id_data(n, debug):
    # birth_year = None
    # birth_month = None
    # birth_date = None

    var = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]

    # 'x'->'X'
    var_id = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']
    n = str(n)
    sum = 0

    for i in range(0, 17):
        sum += int(n[i]) * var[i]
    sum %= 11
    if (var_id[sum]) == str(n[17]):
        # 获得出生年、月、日
        birth_year = n[6:10]
        birth_month = n[10:12]
        birth_date = n[12:14]
        if debug:
            print("身份证号规则核验通过，校验码是：", var_id[sum])
            print("出生于：", n[6:10], "年", n[10:12], "月", n[12:14], "日")

        if year_is_legal(birth_year) and month_is_legal(birth_month) and date_is_legal(birth_year, birth_month,
                                                                                       birth_date):

            return {
                'year': birth_year,
                'month': birth_month,
                'date': birth_date,
            }
        else:
            return None
    else:
        return None


def check_id_code(n, debug):
    if check_id_length(n) and checkX(n):
        results = check_id_data(n, debug)
        if results is not None:
            return results
        else:
            return None
    else:
        return None


if __name__ == '__main__':
    code = input("请输入18位身份证号:")
    print(check_id_code(code, False))
