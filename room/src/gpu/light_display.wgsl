// Apply display controls only after linear-light accumulation, in either mode.
fn display_light(incoming: vec3<f32>, albedo: vec3<f32>, emission: f32,
                 minimum: f32, maximum: f32, emission_scale: f32) -> vec3<f32> {
    let ceiling = max(minimum, select(1.0, maximum, maximum > 0.0));
    let limited = clamp(incoming, vec3<f32>(minimum), vec3<f32>(ceiling));
    let shaded = albedo * limited;
    let glow = albedo * emission * emission_scale;
    let hot = (shaded + glow) / (vec3<f32>(1.0) + glow);
    return mix(shaded, hot, select(0.0, 0.5, emission > 0.0));
}
