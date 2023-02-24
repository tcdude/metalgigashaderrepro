#version 450

in vec2 uv;

out vec4 frag_color;

uniform float i_time;
uniform vec2 i_resolution;
uniform vec2 sg_alpha;
uniform vec2 sg_mouse;

// IQs' magic
float dot2(in vec2 v) {
	return dot(v, v);
}
float dot2(in vec3 v) {
	return dot(v, v);
}
float ndot(in vec2 a, in vec2 b) {
	return a.x * b.x - a.y * b.y;
}

float sd_sphere(vec3 p, float s) {
	return length(p) - s;
}

float sd_box(vec3 p, vec3 b) {
	vec3 q = abs(p) - b;
	return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sd_round_box(vec3 p, vec3 b, float r) {
	vec3 q = abs(p) - b;
	return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

float sd_box_frame(vec3 p, vec3 b, float e) {
	p = abs(p) - b;
	vec3 q = abs(p + e) - e;
	return min(min(length(max(vec3(p.x, q.y, q.z), 0.0)) + min(max(p.x, max(q.y, q.z)), 0.0),
	               length(max(vec3(q.x, p.y, q.z), 0.0)) + min(max(q.x, max(p.y, q.z)), 0.0)),
	           length(max(vec3(q.x, q.y, p.z), 0.0)) + min(max(q.x, max(q.y, p.z)), 0.0));
}

float sd_torus(vec3 p, vec2 t) {
	vec2 q = vec2(length(p.xz) - t.x, p.y);
	return length(q) - t.y;
}

float sd_capped_torus(in vec3 p, in vec2 sc, in float ra, in float rb) {
	p.x = abs(p.x);
	float k = (sc.y * p.x > sc.x * p.y) ? dot(p.xy, sc) : length(p.xy);
	return sqrt(dot(p, p) + ra * ra - 2.0 * ra * k) - rb;
}

float sd_link(vec3 p, float le, float r1, float r2) {
	vec3 q = vec3(p.x, max(abs(p.y) - le, 0.0), p.z);
	return length(vec2(length(q.xy) - r1, q.z)) - r2;
}

float sd_cone(in vec3 p, in vec2 c, float h) {
	// c is the sin/cos of the angle, h is height
	// Alternatively pass q instead of (c,h),
	// which is the point at the base in 2D
	vec2 q = h * vec2(c.x / c.y, -1.0);

	vec2 w = vec2(length(p.xz), p.y);
	vec2 a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
	vec2 b = w - q * vec2(clamp(w.x / q.x, 0.0, 1.0), 1.0);
	float k = sign(q.y);
	float d = min(dot(a, a), dot(b, b));
	float s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y));
	return sqrt(d) * sign(s);
}

float sd_cone_inf(vec3 p, vec2 c) {
	// c is the sin/cos of the angle
	vec2 q = vec2(length(p.xz), -p.y);
	float d = length(q - c * max(dot(q, c), 0.0));
	return d * ((q.x * c.y - q.y * c.x < 0.0) ? -1.0 : 1.0);
}

float sd_plane(vec3 p, vec3 n, float h) {
	// n must be normalized
	return dot(p, n) + h;
}

float sd_hex_prism(vec3 p, vec2 h) {
	const vec3 k = vec3(-0.8660254, 0.5, 0.57735);
	p = abs(p);
	p.xy -= 2.0 * min(dot(k.xy, p.xy), 0.0) * k.xy;
	vec2 d = vec2(length(p.xy - vec2(clamp(p.x, -k.z * h.x, k.z * h.x), h.x)) * sign(p.y - h.x),
	              p.z - h.y);
	return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float sd_tri_prism(vec3 p, vec2 h) {
	vec3 q = abs(p);
	return max(q.z - h.y, max(q.x * 0.866025 + p.y * 0.5, -p.y) - h.x * 0.5);
}

float sd_capsule(vec3 p, vec3 a, vec3 b, float r) {
	vec3 pa = p - a, ba = b - a;
	float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
	return length(pa - ba * h) - r;
}

float sd_vertical_capsule(vec3 p, float h, float r) {
	p.y -= clamp(p.y, 0.0, h);
	return length(p) - r;
}

float sd_capped_cylinder(vec3 p, float h, float r) {
	vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(h, r);
	return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float sd_rounded_cylinder(vec3 p, float ra, float rb, float h) {
	vec2 d = vec2(length(p.xz) - 2.0 * ra + rb, abs(p.y) - h);
	return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - rb;
}

float sd_capped_cone(vec3 p, float h, float r1, float r2) {
	vec2 q = vec2(length(p.xz), p.y);
	vec2 k1 = vec2(r2, h);
	vec2 k2 = vec2(r2 - r1, 2.0 * h);
	vec2 ca = vec2(q.x - min(q.x, (q.y < 0.0) ? r1 : r2), abs(q.y) - h);
	vec2 cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot2(k2), 0.0, 1.0);
	float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
	return s * sqrt(min(dot2(ca), dot2(cb)));
}

float sd_solid_angle(vec3 p, vec2 c, float ra) {
	// c is the sin/cos of the angle
	vec2 q = vec2(length(p.xz), p.y);
	float l = length(q) - ra;
	float m = length(q - c * clamp(dot(q, c), 0.0, ra));
	return max(l, m * sign(c.y * q.x - c.x * q.y));
}

float sd_cut_sphere(vec3 p, float r, float h) {
	// sampling independent computations (only depend on shape)
	float w = sqrt(r * r - h * h);

	// sampling dependant computations
	vec2 q = vec2(length(p.xz), p.y);
	float s = max((h - r) * q.x * q.x + w * w * (h + r - 2.0 * q.y), h * q.x - w * q.y);
	return (s < 0.0) ? length(q) - r : (q.x < w) ? h - q.y : length(q - vec2(w, h));
}

float sd_cut_hollow_sphere(vec3 p, float r, float h, float t) {
	// sampling independent computations (only depend on shape)
	float w = sqrt(r * r - h * h);

	// sampling dependant computations
	vec2 q = vec2(length(p.xz), p.y);
	return ((h * q.x < w * q.y) ? length(q - vec2(w, h)) : abs(length(q) - r)) - t;
}

float sd_death_star(in vec3 p2, in float ra, float rb, in float d) {
	// sampling independent computations (only depend on shape)
	float a = (ra * ra - rb * rb + d * d) / (2.0 * d);
	float b = sqrt(max(ra * ra - a * a, 0.0));

	// sampling dependant computations
	vec2 p = vec2(p2.x, length(p2.yz));
	if (p.x * b - p.y * a > d * max(b - p.y, 0.0))
		return length(p - vec2(a, b));
	else
		return max((length(p) - ra), -(length(p - vec2(d, 0)) - rb));
}

float sd_round_cone(vec3 p, float r1, float r2, float h) {
	// sampling independent computations (only depend on shape)
	float b = (r1 - r2) / h;
	float a = sqrt(1.0 - b * b);

	// sampling dependant computations
	vec2 q = vec2(length(p.xz), p.y);
	float k = dot(q, vec2(-b, a));
	if (k < 0.0) return length(q) - r1;
	if (k > a * h) return length(q - vec2(0.0, h)) - r2;
	return dot(q, vec2(a, b)) - r1;
}

float sd_ellipsoid(vec3 p, vec3 r) {
	float k0 = length(p / r);
	float k1 = length(p / (r * r));
	return k0 * (k0 - 1.0) / k1;
}

float sd_rhombus(vec3 p, float la, float lb, float h, float ra) {
	p = abs(p);
	vec2 b = vec2(la, lb);
	float f = clamp((ndot(b, b - 2.0 * p.xz)) / dot(b, b), -1.0, 1.0);
	vec2 q = vec2(length(p.xz - 0.5 * b * vec2(1.0 - f, 1.0 + f)) *
	                      sign(p.x * b.y + p.z * b.x - b.x * b.y) -
	                  ra,
	              p.y - h);
	return min(max(q.x, q.y), 0.0) + length(max(q, 0.0));
}

float sd_octahedron(vec3 p, float s) {
	p = abs(p);
	float m = p.x + p.y + p.z - s;
	vec3 q;
	if (3.0 * p.x < m)
		q = p.xyz;
	else if (3.0 * p.y < m)
		q = p.yzx;
	else if (3.0 * p.z < m)
		q = p.zxy;
	else
		return m * 0.57735027;

	float k = clamp(0.5 * (q.z - q.y + s), 0.0, s);
	return length(vec3(q.x, q.y - s + k, q.z - k));
}

float sd_pyramid(vec3 p, float h) {
	float m2 = h * h + 0.25;

	p.xz = abs(p.xz);
	p.xz = (p.z > p.x) ? p.zx : p.xz;
	p.xz -= 0.5;

	vec3 q = vec3(p.z, h * p.y - 0.5 * p.x, h * p.x + 0.5 * p.y);

	float s = max(-q.x, 0.0);
	float t = clamp((q.y - 0.5 * p.z) / (m2 + 0.25), 0.0, 1.0);

	float a = m2 * (q.x + s) * (q.x + s) + q.y * q.y;
	float b = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) + (q.y - m2 * t) * (q.y - m2 * t);

	float d2 = min(q.y, -q.x * m2 - q.y * 0.5) > 0.0 ? 0.0 : min(a, b);

	return sqrt((d2 + q.z * q.z) / m2) * sign(max(q.z, -p.y));
}

// 2D

float ud_triangle(vec3 p, vec3 a, vec3 b, vec3 c) {
	vec3 ba = b - a;
	vec3 pa = p - a;
	vec3 cb = c - b;
	vec3 pb = p - b;
	vec3 ac = a - c;
	vec3 pc = p - c;
	vec3 nor = cross(ba, ac);

	return sqrt((sign(dot(cross(ba, nor), pa)) + sign(dot(cross(cb, nor), pb)) +
	                 sign(dot(cross(ac, nor), pc)) <
	             2.0)
	                ? min(min(dot2(ba * clamp(dot(ba, pa) / dot2(ba), 0.0, 1.0) - pa),
	                          dot2(cb * clamp(dot(cb, pb) / dot2(cb), 0.0, 1.0) - pb)),
	                      dot2(ac * clamp(dot(ac, pc) / dot2(ac), 0.0, 1.0) - pc))
	                : dot(nor, pa) * dot(nor, pa) / dot2(nor));
}

float ud_quad(vec3 p, vec3 a, vec3 b, vec3 c, vec3 d) {
	vec3 ba = b - a;
	vec3 pa = p - a;
	vec3 cb = c - b;
	vec3 pb = p - b;
	vec3 dc = d - c;
	vec3 pc = p - c;
	vec3 ad = a - d;
	vec3 pd = p - d;
	vec3 nor = cross(ba, ad);

	return sqrt((sign(dot(cross(ba, nor), pa)) + sign(dot(cross(cb, nor), pb)) +
	                 sign(dot(cross(dc, nor), pc)) + sign(dot(cross(ad, nor), pd)) <
	             3.0)
	                ? min(min(min(dot2(ba * clamp(dot(ba, pa) / dot2(ba), 0.0, 1.0) - pa),
	                              dot2(cb * clamp(dot(cb, pb) / dot2(cb), 0.0, 1.0) - pb)),
	                          dot2(dc * clamp(dot(dc, pc) / dot2(dc), 0.0, 1.0) - pc)),
	                      dot2(ad * clamp(dot(ad, pd) / dot2(ad), 0.0, 1.0) - pd))
	                : dot(nor, pa) * dot(nor, pa) / dot2(nor));
}

// CSG

vec2 smin_cubic(float a, float b, float k) {
	float h = max(k - abs(a - b), 0.0) / k;
	float m = h * h * h * 0.5;
	float s = m * k * (1.0 / 3.0);
	return (a < b) ? vec2(a - s, m) : vec2(b - s, 1.0 - m);
}

vec2 smax_cubic(float a, float b, float k) {
	float h = max(k - abs(a - b), 0.0) / k;
	float m = h * h * h * 0.5;
	float s = m * k * (1.0 / 3.0);
	return (a > b) ? vec2(a - s, m) : vec2(b - s, 1.0 - m);
}

vec4 sdf_union(vec4 a, vec4 b) {
	return (a.a < b.a) ? a : b;
}

vec4 sdf_subtraction(vec4 a, vec4 b) {
	return (-a.a > b.a) ? vec4(a.rgb, -a.a) : b;
}

vec4 sdf_intersection(vec4 a, vec4 b) {
	return (a.a > b.a) ? a : b;
}

vec4 sdf_smooth_union(vec4 a, vec4 b, float k) {
	vec2 sm = smin_cubic(a.a, b.a, k);
	vec3 col = mix(a.rgb, b.rgb, sm.y);
	return vec4(col, sm.x);
}

vec4 sdf_smooth_subtraction(vec4 a, vec4 b, float k) {
	vec2 sm = smax_cubic(-a.a, b.a, k);
	vec3 col = mix(a.rgb, b.rgb, sm.y);
	return vec4(col, sm.x);
}

vec4 sdf_smooth_intersection(vec4 a, vec4 b, float k) {
	vec2 sm = smax_cubic(a.a, b.a, k);
	vec3 col = mix(a.rgb, b.rgb, sm.y);
	return vec4(col, sm.x);
}

// Tools for GLSL 1.10
vec3 dumbround(vec3 v) {
	v += 0.5f;
	return vec3(float(int(v.x)), float(int(v.y)), float(int(v.z)));
}

vec4 render();

void main() {
	frag_color = render() - (i_time * i_resolution.x * 1e-20);
}

////---- shadergen_source ----////
uniform vec4 sg_data[14];
vec4 sg_node4(vec3 p);vec4 sg_node4(vec3 p) {return vec4(sg_data[12][0],sg_data[12][1],sg_data[12][2],sd_box(p,vec3(sg_data[11][1],sg_data[11][2],sg_data[11][3])));}
vec4 get_dist(vec3 p) {vec4 dist=vec4(0.0,0.0,0.0,1000.0);dist = sdf_union(sg_node4(p), dist);return dist;}

#define MAX_STEPS int(sg_data[9][2])
#define MAX_DIST sg_data[9][3]
#define SURF_DIST sg_data[10][0]

#define shininess sg_data[10][1]

vec4 raymarch(vec3 ro, vec3 rd) {
	float dO = 0.0;
	vec3 col = vec3(0.0);
	for (int i = 0; i < MAX_STEPS; i++) {
		vec3 p = ro + rd * dO;
		vec4 dS = get_dist(p);
		dO += dS.a;
		if (abs(dS.a) < SURF_DIST) col = dS.rgb;
		if (dO > MAX_DIST || abs(dS.a) < SURF_DIST) break;
	}
	return vec4(col, dO);
}

vec3 get_normal(vec3 p) {
	const float h = 0.001;
	const vec2 k = vec2(1.0, -1.0);

	return normalize(k.xyy * get_dist(p + k.xyy * h).a + k.yyx * get_dist(p + k.yyx * h).a +
	                 k.yxy * get_dist(p + k.yxy * h).a + k.xxx * get_dist(p + k.xxx * h).a);
}

vec3 get_ray_direction(vec2 nuv, vec3 p, vec3 l, float z) {
	vec3 f = normalize(l - p);
	vec3 r = normalize(cross(vec3(0.0, 1.0, 0.0), f));
	vec3 u = cross(f, r);
	vec3 c = f * z;
	vec3 i = c + nuv.x * r + nuv.y * u;
	vec3 d = normalize(i);
	return d;
}

// https://iquilezles.org/articles/nvscene2008/rwwtt.pdf
float calc_ao(vec3 pos, vec3 nor) {
	float occ = 0.0;
	float sca = 1.0;
	for (int i = 0; i < 5; i++) {
		float h = 0.01 + 0.12 * float(i) / 4.0;
		float d = get_dist(pos + h * nor).a;
		occ += (h - d) * sca;
		sca *= 0.95;
		if (occ > 0.35) break;
	}
	return clamp(1.0 - 3.0 * occ, 0.0, 1.0) * (0.5 + 0.5 * nor.y);
}

mat3 set_camera(in vec3 ro, in vec3 ta, float cr) {
	vec3 cw = normalize(ta - ro);
	vec3 cp = vec3(sin(cr), cos(cr), 0.0);
	vec3 cu = normalize(cross(cw, cp));
	vec3 cv = (cross(cu, cw));
	return mat3(cu, cv, cw);
}

vec3 spherical_harmonics(vec3 n) {
	const float C1 = 0.429043;
	const float C2 = 0.511664;
	const float C3 = 0.743125;
	const float C4 = 0.886227;
	const float C5 = 0.247708;

	const vec3 L00 = vec3(sg_data[2][2], sg_data[2][3], sg_data[3][0]);
	const vec3 L1m1 = vec3(sg_data[3][1], sg_data[3][2], sg_data[3][3]);
	const vec3 L10 = vec3(sg_data[4][0], sg_data[4][1], sg_data[4][2]);
	const vec3 L11 = vec3(sg_data[4][3], sg_data[5][0], sg_data[5][1]);
	const vec3 L2m2 = vec3(sg_data[5][2], sg_data[5][3], sg_data[6][0]);
	const vec3 L2m1 = vec3(sg_data[6][1], sg_data[6][2], sg_data[6][3]);
	const vec3 L20 = vec3(sg_data[7][0], sg_data[7][1], sg_data[7][2]);
	const vec3 L21 = vec3(sg_data[7][3], sg_data[8][0], sg_data[8][1]);
	const vec3 L22 = vec3(sg_data[8][2], sg_data[8][3], sg_data[9][0]);

	return (C1 * L22 * (n.x * n.x - n.y * n.y) + C3 * L20 * n.z * n.z + C4 * L00 - C5 * L20 +
	        2.0 * C1 * L2m2 * n.x * n.y + 2.0 * C1 * L21 * n.x * n.z + 2.0 * C1 * L2m1 * n.y * n.z +
	        2.0 * C2 * L11 * n.x + 2.0 * C2 * L1m1 * n.y + 2.0 * C2 * L10 * n.z) *
	       sg_data[9][1];
}

vec4 render() {
	// vec2 pos = (uv - 0.5) * i_resolution;
	// vec2 nuv = pos / i_resolution.y;
	vec2 frag_coord = uv * i_resolution;

	// camera
	vec3 ta = vec3(sg_data[1][0], sg_data[1][2], sg_data[1][3]);
	// vec3 ro = ta + vec3(0.0, 0.0, -5.0);
	vec3 ro = vec3(sg_data[0][1], sg_data[0][2], sg_data[0][3]);
	// camera-to-world transformation
	mat3 ca = set_camera(ro, ta, 0.0);

	vec3 tot = vec3(0.0);
	int hits = 0;
	for (int m = 0; m < 2; m++)
		for (int n = 0; n < 2; n++) {
			// pixel coordinates
			vec2 o = vec2(float(m), float(n)) / 2.0f - 0.5;
			vec2 frag_pos = (2.0 * (frag_coord + o) - i_resolution.xy) / i_resolution.y;
			vec2 m = (2.0 * (sg_mouse + o) - i_resolution.xy) / i_resolution.y;

			// focal length
			const float fl = sg_data[0][0];

			// ray direction
			vec3 rd = ca * normalize(vec3(frag_pos, fl));

			// ray differentials
			// vec2 px = (2.0 * (fragCoord + vec2(1.0, 0.0)) - iResolution.xy) / iResolution.y;
			// vec2 py = (2.0 * (fragCoord + vec2(0.0, 1.0)) - iResolution.xy) / iResolution.y;
			// vec3 rdx = ca * normalize( vec3(px,fl) );
			// vec3 rdy = ca * normalize( vec3(py,fl) );

			// render
			vec4 rm = raymarch(ro, rd);
			vec3 ambient = vec3(0.05);
			vec3 specular = vec3(1.0);
			vec3 col = vec3(sg_data[1][3], sg_data[2][0], sg_data[2][1]); // vec3(0.0, 1.0, 0.0);

			vec3 l_dir = normalize(vec3(sg_data[10][2], sg_data[10][3], sg_data[11][0]));

			if (rm.a < MAX_DIST) {
				vec3 p = ro + rd * rm.a;
				vec3 n = get_normal(p);
				float occ = calc_ao(p, n);

				float intensity = max(dot(n, l_dir), 0.0);
				vec3 spec = vec3(0.0);

				if (intensity > 0.0) {
					vec3 h = normalize(l_dir - rd);
					float int_spec = max(dot(h, n), 0.0);
					spec = specular * pow(int_spec, shininess);
				}
				rm.rgb += spherical_harmonics(n);
				col = max(intensity * rm.rgb + spec, ambient * rm.rgb) * occ;
				hits++;
				float hl = (0.05 - clamp(length(frag_pos - m), 0.0, 0.05)) / 0.05;
				col.r += hl;
				col.gb = mix(col.gb, vec2(0.0), hl);
			}

			// gamma
			col = pow(col, vec3(0.4545));

			tot += col;
		}
	tot /= 4.0f;
	float alpha = hits > 0 ? sg_alpha.y : sg_alpha.x;
	return vec4(tot, alpha);
}
