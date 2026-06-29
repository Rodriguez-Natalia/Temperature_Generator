import numpy as np

from scipy.stats import gamma

class TemperatureGeneratorWithNoiseStep:
    """Generator of daily temperature serie as a result\
        of point interpolation.
    """
    
    MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    def __init__(self, max_temp_limit=46, min_temp_limit=-2, max_attempts=1000):
        "Initializes decil ranges and temperature limits"
        self.decil_ranges = self._generate_decil_ranges()
        self.max_temp_limit = max_temp_limit
        self.min_temp_limit = min_temp_limit
        self.max_attempts = max_attempts
    
    def _generate_decil_ranges(self):
        """Generates decile intervals (36 in total)
        """
        decil_ranges = []
        day_counter = 0
        decil = 1
        
        for days in self.MONTH_DAYS:
            if days == 31:
                group_sizes = [10, 10, 11]
            elif days == 30:
                group_sizes = [10, 10, 10]
            elif days == 28:
                group_sizes = [10, 10, 8]
        
            start = 0
            for group_size in group_sizes:
                end = start + group_size - 1
                decil_ranges.append((day_counter + start,
                                    day_counter + end, decil))
                start = end + 1
                decil += 1
            
            day_counter += days
        
        return decil_ranges
    
    @staticmethod
    def _linear_interpolation(p1, p2):
        """Determines a straight line between two points.

        Args:
            pi = (day, temperature)
        """
        x1, y1 = p1
        x2, y2 = p2
        
        if x2 == x1:
            return 0, y1
        
        m = (y2-y1) / (x2-x1)
        b = y1 - m * x1
        return m, b
    
    def get_decil(self, day):
        """Find the decile corresponding to a specific day
        """
        for start, end, decil in self.decil_ranges:
            if start <= day <= end:
                return decil
        # Si el día está fuera del rango (365), usar el último decil
        return 36
    
    def generate_spaced_days(self, spacing):
        """Generate an array of days spaced to cover exactly 365 days.
        
        Args:
            spacing : Gap between days (2, 3, 5, etc.).
            
        Returns:
            Days from 0 to 364 with the specified spacing.
        """
        # Calcular cuántos puntos necesitamos para cubrir 365 días
        n_points = int(np.ceil(365 / spacing))
        
        # Generar días desde 0 hasta el último punto que no exceda 364
        days = np.arange(0, n_points * spacing, spacing)
        
        # Asegurar que el último día sea 364 (día 365 sería índice 364)
        if days[-1] > 364:
            days = days[days <= 364]
        
        # Si el último día no es 364, agregarlo
        if days[-1] != 364:
            days = np.append(days, 364)
        
        return days
    
    def _validate_temperature_limits(self, max_value, min_value):
        """Check that the temperatures are within the permitted limits.
        
        Args:
            max_value : Maximum temperature.
            min_value : Minimum temperature.
            
        Returns:
            bool: True if they are within the permitted limits.
        """
        return (max_value <= self.max_temp_limit and 
                min_value >= self.min_temp_limit)
    
    def _apply_temperature_limits(self, temperature, temp_type=None):
        """Apply temperature limits to a single value.
        
        Args:
            temperature : Temperature at which the limit applies.
            temp_type : 'max', 'min', or None for both limits.
            
        Returns:
            float: Temperature within the limits.
        """
        if temp_type == 'max':
            return min(temperature, self.max_temp_limit)
        elif temp_type == 'min':
            return max(temperature, self.min_temp_limit)
        else:
            return np.clip(temperature, self.min_temp_limit, self.max_temp_limit)
    
    def generate_daily_temperature_with_noise(self, points, std_params, std_max, std_min,
                                              temp_type, reference_series=None):
        """Generates a daily temperature series by linear interpolation with noise applied only to intermediate points.

        Args:
            points : Array of points [[days, temperature], ...]
            std_params: Standard deviation parameters per decade.
            std_max: Standard deviation for maximum temperature per decade.
            std_min: Standard deviation for minimum temperature per decade.
            shape_params: Shape parameters for gamma distribution.
            scale_params: Scale parameters for gamma distribution.
            temp_type: 'mean', 'max', or 'min'
            reference_series: Reference series for checking conditions (tmean for tmax/tmin)
        
        Returns:
            Array with daily temperatures (365 days).
        """
        if len(points) < 2:
            raise ValueError("Must be at least two points")
    
        daily_temp = []
        
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i+1]
            x1, x2 = p1[0], p2[0]
            
            m, b = self._linear_interpolation(p1, p2)
            
            n_points = int((x2 - x1))
            xs = np.linspace(x1, x2, n_points, endpoint=False)
            ys = m * xs + b
            
            # Aplicar ruido solo a los puntos intermedios (no a los puntos originales)
            noisy_ys = []
            for j, (x, y) in enumerate(zip(xs, ys)):
                day = int(round(x))
                decil = self.get_decil(day)
                idx = decil - 1
                
                # Solo aplicar ruido si no es un punto original
                is_original_point = (x == x1 and j == 0) or (x == x2 and j == len(xs)-1)
                
                if not is_original_point:
                    if temp_type == 'mean':
                        # Ruido normal para temperatura media
                        noise = np.random.normal(0, std_params[idx])
                        temp_candidate = y + noise
                        # Aplicar límites a temperatura media
                        temp_candidate = self._apply_temperature_limits(temp_candidate)
                        noisy_ys.append(temp_candidate)
                    
                    elif temp_type == 'max':
                        # Ruido normal para temperatura máxima
                        for attempt in range(self.max_attempts):
                            noise = np.random.normal(0, std_max[idx])
                            temp_candidate = y + noise
                            
                            # Aplicar límite máximo
                            temp_candidate = self._apply_temperature_limits(temp_candidate, 'max')
                            
                            # Verificar que tmax > tmean
                            if reference_series is not None and day < len(reference_series):
                                if temp_candidate > reference_series[day]:
                                    noisy_ys.append(temp_candidate)
                                    break
                            else:
                                # Si no hay serie de referencia, aceptar cualquier valor
                                noisy_ys.append(temp_candidate)
                                break
                            
                            # Si llegamos al último intento, usar el valor aunque no cumpla
                            if attempt == self.max_attempts - 1:
                                if reference_series is not None and day < len(reference_series):
                                    enforced_temp = y + 0.5 #max(temp_candidate, reference_series[day] + 0.5) #0.1
                                    enforced_temp = self._apply_temperature_limits(enforced_temp, 'max')
                                    noisy_ys.append(enforced_temp)
                                else:
                                    temp_candidate = self._apply_temperature_limits(temp_candidate, 'max')
                                    noisy_ys.append(temp_candidate)
                    
                    elif temp_type == 'min':
                        # Ruido normal para temperatura mínima
                        for attempt in range(self.max_attempts):
                            noise = np.random.normal(0, std_min[idx])
                            temp_candidate = y + noise
                            
                            # Aplicar límite mínimo
                            temp_candidate = self._apply_temperature_limits(temp_candidate, 'min')
                            
                            # Verificar que tmin < tmean
                            if reference_series is not None and day < len(reference_series):
                                if temp_candidate < reference_series[day]:
                                    noisy_ys.append(temp_candidate)
                                    break
                            else:
                                # Si no hay serie de referencia, aceptar cualquier valor
                                noisy_ys.append(temp_candidate)
                                break
                            
                            # Si llegamos al último intento, usar el valor aunque no cumpla
                            if attempt == self.max_attempts - 1:
                                if reference_series is not None and day < len(reference_series):
                                    enforced_temp = y - 0.5 #min(temp_candidate, reference_series[day] - 0.5)
                                    enforced_temp = self._apply_temperature_limits(enforced_temp, 'min')
                                    noisy_ys.append(enforced_temp)
                                else:
                                    temp_candidate = self._apply_temperature_limits(temp_candidate, 'min')
                                    noisy_ys.append(temp_candidate)
                else:
                    # Mantener el punto original sin ruido, pero aplicar límites
                    original_temp = self._apply_temperature_limits(y, temp_type)
                    noisy_ys.append(original_temp)
            
            daily_temp.append(noisy_ys)
        
        # Agregar el último punto original (sin ruido, pero con límites aplicados)
        last_temp = self._apply_temperature_limits(points[-1][1], temp_type)
        daily_temp.append([last_temp])
        
        full_series = np.concatenate(daily_temp)
        
        # Asegurar que tenemos exactamente 365 días (índices 0-364)
        return full_series[:365]
    
    def _generate_temperature_point_with_limits(self, day, mean_params, std_params, 
                                               shape_params, scale_params):
        """Genera un punto de temperatura con validación de límites.
        
        Args:
            day : Day of the year.
            mean_params: Mean temperature parameters per decade.
            std_params: Standard deviation parameters per decade.
            shape_params : Shape parameters for gamma distribution.
            scale_params : Scale parameters for gamma distribution.
            
        Returns:
            tuple: (mean_value, max_value, min_value)
        """
        decil = self.get_decil(day)
        idx = decil - 1
        
        for attempt in range(self.max_attempts):
            # Generar valores de temperatura
            mean_value = np.random.normal(mean_params[idx], std_params[idx])
            variations = gamma.rvs(a=shape_params[idx], loc=0, scale=scale_params[idx])
            max_value = mean_value + variations
            min_value = mean_value - variations
            
            # Aplicar límites
            max_value = self._apply_temperature_limits(max_value, 'max')
            min_value = self._apply_temperature_limits(min_value, 'min')
            
            # Validar que max > min (consistencia física)
            if max_value > min_value:
                # Recalcular mean_value para mantener consistencia después de aplicar límites
                adjusted_mean = (max_value + min_value) / 2
                return adjusted_mean, max_value, min_value
        
        # Si no se encontraron valores válidos después de max_attempts
        # Usar valores por defecto dentro de los límites
        default_mean = (self.max_temp_limit + self.min_temp_limit) / 2
        default_max = self.max_temp_limit - 1
        default_min = self.min_temp_limit + 1
        
        return default_mean, default_max, default_min
    
    def generate_temperature_points(self, spacing, mean_params, std_params, std_max, std_min,
                                    shape_params, scale_params):
        """Generates temperature points for specific days according\
            to the decade they belong to.
            
        Args:
            spacing : gap between days (2, 3, 5, etc.)
            mean_params : Mean temperature parameters per decade.
            std_params : Standard deviation parameters per decade.
            std_max: Standard deviation for maximum temperature per decade.
            std_min: Standard deviation for minimum temperature per decade.
            shape_params : Shape parameters for gamma distribution.
            scale_params : Scale parameters for gamma distribution.
        Returns:
            Arrays with 365 points.
        """
        # Generar días espaciados
        days_interval = self.generate_spaced_days(spacing)
        
        # Paso 1: Generar puntos iniciales para los días del intervalo CON LÍMITES
        p_tmean_initial = []
        p_tmax_initial = []
        p_tmin_initial = []
        
        for day in days_interval:
            mean_value, max_value, min_value = self._generate_temperature_point_with_limits(
                day, mean_params, std_params, shape_params, scale_params)
            
            p_tmean_initial.append([day, mean_value])
            p_tmax_initial.append([day, max_value])
            p_tmin_initial.append([day, min_value])
        
        # Paso 2: Primero generar serie de tmean (sin dependencias)
        t_tmean_final = self.generate_daily_temperature_with_noise(
            p_tmean_initial, std_params, std_max, std_min, 'mean')
        
        # Paso 3: Generar tmax y tmin usando tmean como referencia
        t_tmax_final = self.generate_daily_temperature_with_noise(
            p_tmax_initial, std_params, std_max, std_min, 
            'max', t_tmean_final)
        
        t_tmin_final = self.generate_daily_temperature_with_noise(
            p_tmin_initial, std_params, std_max, std_min,
            'min', t_tmean_final)
        
        # Verificación final de límites
        final_max = np.max(t_tmax_final)
        final_min = np.min(t_tmin_final)
        
        if final_max > self.max_temp_limit or final_min < self.min_temp_limit:
            print(f"WARNING: Limits exceeded after interpolation.")
            print(f"Max: {final_max:.2f}°C, Limit: {self.max_temp_limit}°C")
            print(f"Min: {final_min:.2f}°C, Limit: {self.min_temp_limit}°C")
        
        return t_tmin_final, t_tmax_final, t_tmean_final
