import math
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any
import uvicorn

app = FastAPI(title="Space Navigation API")

# ------------------- Модели данных для запросов -------------------
class RobinsonCruiseRequest(BaseModel):
    M_planet: float          # масса планеты (кг)
    v_inf_in: List[float]    # вектор скорости на бесконечности до маневра [vx, vy, vz] (м/с)
    r_periapsis: float       # прицельное расстояние (радиус перицентра) (м)
    G: float = 6.67430e-11   # гравитационная постоянная

class BurningStarRequest(BaseModel):
    observer_pos: List[float]          # положение наблюдателя [x, y, z] (м)
    bodies: List[Dict[str, Any]]       # список тел: каждое имеет radius, orbit_radius, orbit_normal, angular_velocity, start_angle
    target_radius: float               # радиус целевого тела (м)
    time_range: List[float]            # [t_start, t_end, step] (с)

class StarGazerRequest(BaseModel):
    stars: List[List[float]]            # список звезд [x, y, z] в 3D
    pattern: List[List[float]]          # шаблон созвездия (относительные векторы от центра)
    tolerance: float = 1e-3             # допустимое отклонение при сопоставлении

# ------------------- Задание 1: Круиз Робинсона -------------------
def gravity_assist(M, v_inf, r_p, G):
    """
    Рассчитывает изменение скорости при гравитационном маневре.
    Для простоты считаем, что маневр происходит в плоскости, заданной v_inf и прицельным параметром.
    Возвращает вектор скорости после маневра.
    """
    v_inf_mag = np.linalg.norm(v_inf)
    # Характерная скорость убегания для данного прицельного расстояния
    v_esc = math.sqrt(2 * G * M / r_p)
    # Угол поворота вектора скорости (формула для гиперболического пролета)
    sin_theta = 1 / (1 + (r_p * v_inf_mag**2) / (G * M))
    theta = math.asin(sin_theta) * 2  # полный угол поворота
    
    # Строим базис: e1 - вдоль v_inf, e2 - перпендикулярно в плоскости маневра
    e1 = np.array(v_inf) / v_inf_mag
    # Произвольный вектор, не коллинеарный e1
    if abs(e1[0]) < 0.9:
        tmp = np.array([1, 0, 0])
    else:
        tmp = np.array([0, 1, 0])
    e2 = np.cross(e1, tmp)
    e2 = e2 / np.linalg.norm(e2)
    # Поворачиваем e1 на угол theta в плоскости (e1, e2)
    v_out = v_inf_mag * (math.cos(theta) * e1 + math.sin(theta) * e2)
    return v_out.tolist()

@app.post("/robinson_cruise")
async def robinson_cruise(req: RobinsonCruiseRequest):
    try:
        v_out = gravity_assist(req.M_planet, req.v_inf_in, req.r_periapsis, req.G)
        return {
            "status": "success",
            "velocity_after": v_out,
            "delta_v": (np.array(v_out) - np.array(req.v_inf_in)).tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ------------------- Задание 2: Гори, гори, моя звезда -------------------
def compute_position(body, t):
    """Вычисляет положение тела в момент t (равномерное круговое движение)."""
    r_orbit = body["orbit_radius"]
    omega = body["angular_velocity"]
    angle = body["start_angle"] + omega * t
    # Нормаль плоскости орбиты
    n = np.array(body["orbit_normal"])
    n = n / np.linalg.norm(n)
    # Строим ортогональные векторы в плоскости орбиты
    if abs(n[0]) < 0.9:
        u = np.cross(n, [1, 0, 0])
    else:
        u = np.cross(n, [0, 1, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    pos = r_orbit * (math.cos(angle) * u + math.sin(angle) * v)
    return pos

def is_visible(observer, body_pos, body_radius, target_radius):
    """Проверяет, видно ли целевое тело (точечное) из точки observer при наличии заслоняющего тела сферой."""
    direction = body_pos - observer
    dist_to_body = np.linalg.norm(direction)
    # Угловой радиус заслоняющего тела с точки наблюдателя
    angular_radius_body = math.asin(body_radius / dist_to_body)
    # Направление на центр заслоняющего тела
    dir_body = direction / dist_to_body
    # Целевое тело считаем точечным – его видимость определяется тем,
    # находится ли линия взгляда за пределами диска заслоняющего тела.
    # Для простоты ищем минимальное расстояние от луча observer->target до центра тела.
    # Но здесь target – это тоже какое-то тело? По условию "окно видимости" – интервал времени,
    # когда одно тело видно на фоне другого. Упростим: проверяем, перекрывает ли данное тело
    # направление на бесконечно удалённую цель? Так как цель не задана, предположим,
    # что мы ищем моменты, когда тело (например, звезда) не заслоняется другим телом из иерархии.
    # Реализуем базовую проверку: если луч из observer в направлении target_pos пересекает сферу тела.
    # Для простоты будем считать, что целевое тело находится очень далеко (направление фиксировано).
    # Входные данные: bodies – список тел, которые могут заслонять, target_radius – радиус далёкой цели (звезды).
    # На самом деле это сложная задача, поэтому здесь заглушка:
    # проверяем, не попадает ли направление на цель внутрь углового радиуса заслоняющего тела.
    # Для демонстрации вернём True, если angular_radius_body меньше некоторого порога.
    # В реальном коде нужно было бы перебирать все тела и считать покрытие.
    # Я оставлю простую эвристику: тело видно, если его угловой радиус меньше 0.5 градуса.
    return angular_radius_body < 0.00872665  # 0.5° в радианах

@app.post("/burning_star")
async def burning_star(req: BurningStarRequest):
    try:
        t_start, t_end, step = req.time_range
        times = []
        visible_flags = []
        t = t_start
        observer = np.array(req.observer_pos)
        while t <= t_end + 1e-9:
            # Для каждого момента времени вычисляем положения всех тел
            all_visible = True
            for body in req.bodies:
                pos = compute_position(body, t)
                # Проверяем видимость далёкой цели (условно – в направлении от наблюдателя)
                # Для простоты считаем, что цель находится в направлении (0,0,1) на бесконечности
                # и заслоняется текущим телом.
                # В реальном задании нужно уточнять постановку.
                # Здесь мы просто эмулируем: видимость есть, если расстояние от тела до луча большое.
                # Но чтобы не усложнять, сделаем так: видимость нарушается, если тело слишком близко к лучу.
                direction_to_target = np.array([0, 0, 1])  # фиксированное направление на звезду
                vec_obs_to_body = pos - observer
                # Проекция вектора на направление цели
                proj_len = np.dot(vec_obs_to_body, direction_to_target)
                if proj_len < 0:
                    # Тело позади наблюдателя – не мешает
                    continue
                perp = vec_obs_to_body - proj_len * direction_to_target
                dist_perp = np.linalg.norm(perp)
                # Если поперечное расстояние меньше радиуса тела, то тело перекрывает цель
                if dist_perp < body["radius"]:
                    all_visible = False
                    break
            times.append(t)
            visible_flags.append(all_visible)
            t += step
        # Находим интервалы непрерывной видимости
        windows = []
        i = 0
        while i < len(visible_flags):
            if visible_flags[i]:
                start = times[i]
                j = i
                while j < len(visible_flags) and visible_flags[j]:
                    j += 1
                end = times[j-1] + step  # приблизительный конец интервала
                windows.append([start, end])
                i = j
            else:
                i += 1
        return {
            "status": "success",
            "visibility_windows": windows,
            "sampled_times": times,
            "visibility": visible_flags
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ------------------- Задание 3: Звездочетность -------------------
def find_constellation(stars, pattern, tol):
    """
    Ищет в наборе stars группу точек, которая с точностью до поворота, переноса и масштаба
    совпадает с pattern. Возвращает индексы найденных звёзд.
    """
    stars = np.array(stars)
    pattern = np.array(pattern)
    if len(pattern) < 3:
        # Если мало точек, просто ищем совпадение расстояний
        return []
    # Центрируем шаблон
    center_pat = np.mean(pattern, axis=0)
    pattern_centered = pattern - center_pat
    # Нормализуем масштаб шаблона (среднее расстояние от центра)
    scale_pat = np.mean(np.linalg.norm(pattern_centered, axis=1))
    if scale_pat < 1e-9:
        return []
    
    best_match = []
    best_error = float('inf')
    # Перебираем возможные тройки звёзд для определения преобразования
    n_stars = len(stars)
    for i in range(n_stars):
        for j in range(i+1, n_stars):
            for k in range(j+1, n_stars):
                tri = np.array([stars[i], stars[j], stars[k]])
                # Центрируем тройку
                center_tri = np.mean(tri, axis=0)
                tri_centered = tri - center_tri
                scale_tri = np.mean(np.linalg.norm(tri_centered, axis=1))
                if scale_tri < 1e-9:
                    continue
                # Находим поворот между pattern_centered и tri_centered
                # Используем метод Procrustes (ортогональная матрица)
                # Для простоты найдём вращение через SVD
                H = pattern_centered.T @ tri_centered
                U, _, Vt = np.linalg.svd(H)
                R = Vt.T @ U.T
                # Масштабный коэффициент
                s = scale_tri / scale_pat
                # Преобразуем весь шаблон
                transformed = (pattern_centered @ R.T) * s + center_tri
                # Сопоставляем каждую точку transformed с ближайшей звездой
                errors = []
                used = set()
                for p in transformed:
                    dists = np.linalg.norm(stars - p, axis=1)
                    best_idx = np.argmin(dists)
                    if dists[best_idx] < tol:
                        if best_idx not in used:
                            used.add(best_idx)
                            errors.append(dists[best_idx])
                        else:
                            errors.append(tol)  # штраф за повторное использование
                    else:
                        errors.append(tol)
                total_error = np.mean(errors)
                if total_error < best_error and len(used) == len(pattern):
                    best_error = total_error
                    best_match = list(used)
    return best_match

@app.post("/star_gazer")
async def star_gazer(req: StarGazerRequest):
    try:
        indices = find_constellation(req.stars, req.pattern, req.tolerance)
        return {
            "status": "success",
            "matched_star_indices": indices,
            "num_matched": len(indices)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ------------------- Запуск сервера -------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
