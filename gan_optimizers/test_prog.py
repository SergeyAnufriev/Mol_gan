name = input("Введите Ваше имя и фамилию: ")
age = int(input("Введите Ваш возраст: "))
weight = int(input("Введите Ваш вес: "))


if weight >50 and weight <120:
    if age < 30:
        print('vse ok')
    else:
        print('na vash vkus')
else:
    if age>30 and age <40:
        print('zanytsya soboi')
    elif age>40:
        print('k vrachu')
    else:
        print('na vash vkus')




