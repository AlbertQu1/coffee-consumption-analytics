#datos de la compra realizada
costo_maquina = 12239
compras_cafe = [
    {'gramos': 500, 'costo': 120, 'tazas': 22},
    {'gramos': 500, 'costo': 192.50, 'tazas': 28},
    {'gramos': 500, 'costo': 375, 'tazas': 23},
    {'gramos': 100, 'costo': 50, 'tazas': 6},
    {'gramos': 500, 'costo': 420, 'tazas': 0}
]
precio_taza_cafeteria = 60

#calculo del consumo de cafe
total_cafe_gastado = 0
total_tazas = 0

for compra in compras_cafe:
    total_cafe_gastado += compra['costo']
    total_tazas += compra['tazas']

#inversion total
inversion_total= costo_maquina + total_cafe_gastado

#csoto promedio por taza echas en casa
costo_promedio_taza = total_cafe_gastado / total_tazas

#ahorro por taza
ahorro_por_taza = precio_taza_cafeteria - costo_promedio_taza

#ahorro acumulado
ahorro_acumulado = ahorro_por_taza * total_tazas

#punto de equilibrio
punto_de_equilibrio = inversion_total /ahorro_por_taza

print(f'Costo promedio por taza en casa: ${costo_promedio_taza:.2f} MXN')
print(f'Ahorro por taza vs cafetería: ${ahorro_por_taza:.2f} MXN')
print(f'Ahorro total acumulado: ${ahorro_acumulado:.2f} MXN')
print(f'Tazas necesarias para el punto de equilibrio: {punto_de_equilibrio:.0f}')
print(f'Tazas consumidas al dia de hoy {total_tazas}')

print('\nTiempo para recuperar la inversión:')
for tazas_dia in [1, 2, 3, 4]:
    dias = punto_de_equilibrio / tazas_dia
    meses = dias /30
    print(f'-  {tazas_dia} tazas por día: {dias:.0f} días(~{meses:.1f} meses)')


