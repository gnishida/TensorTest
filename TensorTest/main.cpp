#include <glm/glm.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <iostream>
#include <list>

#ifndef M_PI
#define M_PI	3.14159265358
#endif

double delta = 1e-6f;

/**
 * 座標pt、角度angleから開始して１本の道路を生成する。
 *
 * @param size						ターゲットエリアの一辺の長さ
 * @param angle						角度
 * @param pt						開始位置
 * @param curvature					曲率
 * @param segment_length			segment length
 * @param regular_elements [OUT]	tensor fieldを指定するためのコントロール情報
 * @param vertices [OUT]			生成された頂点をこのリストに追加する
 * @param edges	[OUT]				生成されたエッジをこのリストに追加する
 */
void growRoads(int size, double angle, glm::dvec2 pt, double curvature, double segment_length, std::vector<std::pair<glm::dvec2, double>>& regular_elements, std::vector<std::pair<glm::dvec2, int>>& vertices, std::vector<std::pair<glm::dvec2, glm::dvec2>>& edges) {
	int num_steps = 5;

	while (true) {
		// 今後５ステップ分の曲率を、平均がcurvatureになるようランダムに決定する。
		std::vector<double> rotates;
		double total = 0.0f;
		for (int i = 0; i < num_steps; ++i) {
			double r = rand() % 100;
			rotates.push_back(r);
			total += r;
		}
		for (int i = 0; i < num_steps; ++i) {
			rotates[i] = rotates[i] / total * curvature * num_steps;
		}
		
		// 曲がる方向を決定する
		int rotate_dir = rand() % 2 == 0 ? 1 : -1;

		// 5ステップ分の道路セグメントを生成する
		for (int i = 0; i < num_steps; ++i) {
			// ターゲットエリア外なら終了
			if (pt.x < -size / 2 || pt.x > size / 2 || pt.y < -size / 2 || pt.y > size / 2) return;

			angle += rotates[i] * rotate_dir;
			glm::dvec2 pt2 = pt + glm::dvec2(cos(angle), sin(angle)) * segment_length;
			vertices.push_back(std::make_pair(pt2, 1));
			edges.push_back(std::make_pair(pt, pt2));
			regular_elements.push_back(std::make_pair(pt2, angle));

			pt = pt2;
		}		
	}
}

bool isIntersect(const glm::dvec2& a, const glm::dvec2& b, const glm::dvec2& c, const glm::dvec2& d, glm::dvec2& intPt) {
	glm::dvec2 u = b - a;
	glm::dvec2 v = d - c;

	double numer = v.x * (c.y - a.y) + v.y * (a.x - c.x);
	double denom = u.y * v.x - u.x * v.y;

	if (denom == 0.0)  {
		return false;
	}

	double t0 = numer / denom;

	glm::dvec2 ipt = a + t0*u;
	glm::dvec2 tmp = ipt - c;
	double t1;
	if (glm::dot(tmp, v) > 0.0) {
		t1 = glm::length(tmp) / glm::length(v);
	}
	else {
		t1 = -1.0f * glm::length(tmp) / glm::length(v);
	}

	//Check if intersection is within segments
	if (!(t0 >= delta && t0 <= 1.0 - delta && t1 >= delta && t1 <= 1.0 - delta)) {
		return false;
	}

	glm::dvec2 dirVec = b - a;
	intPt = a + t0 * dirVec;

	return true;
}

/**
 * 指定されたtensor filedに基づいて、ptからsegment_length分の道路セグメントを生成する。
 * ただし、ターゲットエリア外に出るか、既存セグメントとの交差点が既存交差点の近くなら、途中で終了する。
 *
 * @param tensor				tensor field
 * @param segment_length		segment length
 * @param near_threshold		near threshold
 * @param pt [OUT]				この座標から道路セグメント生成を開始する
 * @param type					1 -- major eigen vector / 2 -- minor eigen vector
 * @param dir					1 -- 正の方向 / -1 -- 負の方向
 * @param new_vertices [OUT]	既存セグメントとの交差点をこのリストに追加する
 * @param vertices			既存交差点
 * @param edges	[OUT]			既存セグメント
 * @return		0 -- 正常終了 / 1 -- ターゲットエリア外に出て終了 / 2 -- 既存交差点の近くで交差して終了
 */
int generateRoadSegmentByTensor(const cv::Mat& tensor, double segment_length, double near_threshold, glm::dvec2& pt, int type, int dir, std::vector<std::pair<glm::dvec2, int>>& new_vertices, const std::vector<std::pair<glm::dvec2, int>>& vertices, std::vector<std::pair<glm::dvec2, glm::dvec2>>& edges) {
	int num_step = 5;
	double step_length = segment_length / num_step;

	int result = 0;

	for (int i = 0; i < num_step; ++i) {
		int c = pt.x + tensor.cols / 2;
		int r = pt.y + tensor.rows / 2;
		if (c < 0 || c >= tensor.cols || r < 0 || r >= tensor.rows) {
			result = 1;
			break;
		}

		/////////////////////////////////////////////////////////////////////
		// use Runge-Kutta 4 to obtain the next angle
		double angle1 = tensor.at<double>(r, c);		
		if (type == 2) angle1 += M_PI / 2;	// minor eigen vectorならPI/2を足す

		// angle2
		glm::vec2 pt2 = pt + glm::dvec2(cos(angle1), sin(angle1)) * (step_length * 0.5 * dir);
		int c2 = pt2.x + tensor.cols / 2;
		int r2 = pt2.y + tensor.rows / 2;
		double angle2 = angle1;
		if (c2 >= 0 && c2 < tensor.cols && r2 >= 0 && r2 < tensor.rows) {
			angle2 = tensor.at<double>(r2, c2);
			if (type == 2) angle2 += M_PI / 2;	// minor eigen vectorならPI/2を足す
		}

		// angle3
		glm::vec2 pt3 = pt + glm::dvec2(cos(angle2), sin(angle2)) * (step_length * 0.5 * dir);
		int c3 = pt3.x + tensor.cols / 2;
		int r3 = pt3.y + tensor.rows / 2;
		double angle3 = angle2;
		if (c3 >= 0 && c3 < tensor.cols && r3 >= 0 && r3 < tensor.rows) {
			angle3 = tensor.at<double>(r3, c3);
			if (type == 2) angle3 += M_PI / 2;	// minor eigen vectorならPI/2を足す
		}

		// angle4
		glm::vec2 pt4 = pt + glm::dvec2(cos(angle3), sin(angle3)) * (step_length * dir);
		int c4 = pt4.x + tensor.cols / 2;
		int r4 = pt4.y + tensor.rows / 2;
		double angle4 = angle3;
		if (c4 >= 0 && c4 < tensor.cols && r4 >= 0 && r4 < tensor.rows) {
			angle4 = tensor.at<double>(r4, c4);
			if (type == 2) angle4 += M_PI / 2;	// minor eigen vectorならPI/2を足す
		}

		// RK4によりangleを計算
		double angle = angle1 / 6.0 + angle2 / 3.0 + angle3 / 3.0 + angle4 / 6.0;

		// 次のステップの座標を求める
		glm::vec2 next_pt = pt + glm::dvec2(cos(angle), sin(angle)) * (step_length * dir);

		// 交差点を求める
		for (int k = 0; k < edges.size(); ++k) {
			glm::dvec2 intPt;
			if (isIntersect(edges[k].first, edges[k].second, pt, next_pt, intPt)) {
				new_vertices.push_back(std::make_pair(intPt, 3));

				// 既に近くに頂点がないかチェック
				bool near = false;
				for (int k = 0; k < vertices.size(); ++k) {
					// major vector と minor vectorで異なる場合は、チェックしない
					if (!(vertices[k].second & type)) continue;

					if (glm::length(vertices[k].first - intPt) < near_threshold) {
						near = true;
						break;
					}
				}

				if (near) {
					// 道路セグメントの生成を交差点でストップさせる
					next_pt = intPt;
					i = num_step;
					result = 2;
					break;
				}
			}
		}

		edges.push_back(std::make_pair(pt, next_pt));

		pt = next_pt;
	}

	return result;
}

/**
 * 指定されたtensor fieldに基づいて、ptから道路生成を行う。
 *
 * @param tensor			tensor field
 * @param segment_length	segment length
 * @param near_threshold	near threshold
 * @param pt				この座標から道路セグメント生成を開始する
 * @param type				1 -- major eigen vector / 2 -- minor eigen vector
 * @param dir				1 -- 正の方向 / -1 -- 負の方向
 * @param vertices [OUT]	既存交差点
 * @param edges	[OUT]		既存セグメント
 * @param seeds	[OUT]		新規シードをこのリストに追加する
 */
void generateRoadByTensor(const cv::Mat& tensor, double segment_length, double near_threshold, glm::dvec2 pt, int type, int dir, std::vector<std::pair<glm::dvec2, int>>& vertices, std::vector<std::pair<glm::dvec2, glm::dvec2>>& edges, std::list<std::pair<glm::dvec2, int>>& seeds) {
	std::vector<std::pair<glm::dvec2, int>> new_vertices;

	while (true) {
		int result = generateRoadSegmentByTensor(tensor, segment_length, near_threshold, pt, type, dir, new_vertices, vertices, edges);
		if (result == 1) {	// ターゲットエリア外に出た
			break;
		}
		else if (result == 2) {	// 既存交差点近くのエッジに交差した
			break;
		}
		else {
			// 既に近くに頂点がないかチェック
			bool near = false;
			for (int k = 0; k < vertices.size(); ++k) {
				// major vector と minor vectorで異なる場合は、チェックしない
				if (!(vertices[k].second & type)) continue;

				if (glm::length(vertices[k].first - pt) < near_threshold) near = true;
			}

			if (near) break;
		}

		new_vertices.push_back(std::make_pair(pt, type));
		seeds.push_back(std::make_pair(pt, 3 - type));
	}

	// 新規交差点をverticesに追加
	vertices.insert(vertices.end(), new_vertices.begin(), new_vertices.end());
}

/**
 * 指定されたtensor fieldに基づいて道路網を生成する。
 *
 * @param tensor			tensor field
 * @param segment_length	segment length
 * @param near_threshold	near threshold
 * @param vertices [OUT]	既存交差点
 * @param edges	[OUT]		既存セグメント
 */
void generateRoadsByTensor(const cv::Mat& tensor, double segment_length, double near_threshold, std::vector<std::pair<glm::dvec2, int>>& vertices, std::vector<std::pair<glm::dvec2, glm::dvec2>>& edges) {
	std::list<std::pair<glm::dvec2, int>> seeds;
	for (int i = 0; i < vertices.size(); ++i) {
		seeds.push_back(std::make_pair(vertices[i].first, 3 - vertices[i].second));
	}

	int count = 0;
	while (!seeds.empty()) {
		glm::dvec2 pt = seeds.front().first;
		int type = seeds.front().second;
		seeds.pop_front();

		if (pt.x < -tensor.cols / 2 || pt.x >= tensor.cols / 2 || pt.y < -tensor.rows / 2 || pt.y >= tensor.rows / 2) continue;

		// 既に近くに頂点がないかチェック
		bool near = false;
		for (int k = 0; k < vertices.size(); ++k) {
			// major vector と minor vectorで異なる場合は、チェックしない
			if (!(vertices[k].second & type)) continue;

			double dist = glm::length(vertices[k].first - pt);
			if (dist < near_threshold) near = true;
		}

		std::cout << count << ": (" << pt.x << ", " << pt.y << ") " << type << (near ? " canceled" : "") << std::endl;
		
		if (!near) {
			generateRoadByTensor(tensor, segment_length, near_threshold, pt, type, 1, vertices, edges, seeds);
			generateRoadByTensor(tensor, segment_length, near_threshold, pt, type, -1, vertices, edges, seeds);
		}

		count++;
		if (count > 500) break;
	}

}

void saveTensorImage(const cv::Mat& tensor, const std::string& filename) {
	cv::Mat result(tensor.size(), CV_8U, cv::Scalar(255));
	for (int r = 0; r < tensor.rows; r += 50) {
		for (int c = 0; c < tensor.cols; c += 50) {
			int x1 = c;
			int y1 = r;
			double angle = tensor.at<double>(r, c);
			int x2 = x1 + cos(angle) * 30;
			int y2 = y1 + sin(angle) * 30;
			cv::line(result, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0), 1, cv::LINE_AA);
		}
	}

	cv::flip(result, result, 0);
	cv::imwrite(filename.c_str(), result);
}

void saveRoadImage(int size, const std::vector<std::pair<glm::dvec2, int>>& vertices, const std::vector<std::pair<glm::dvec2, glm::dvec2>>& edges, const std::string& filename, bool showVertices, bool showArrow) {
	cv::Mat result(size, size, CV_8UC3, cv::Scalar(255, 255, 255));
	for (int i = 0; i < edges.size(); ++i) {
		double x1 = edges[i].first.x + size / 2;
		double y1 = edges[i].first.y + size / 2;
		double x2 = edges[i].second.x + size / 2;
		double y2 = edges[i].second.y + size / 2;

		cv::line(result, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

		if (showArrow) {
			double angle = atan2(y2 - y1, x2 - x1);
			double angle1 = angle + M_PI - 20.0 / 180.0 * M_PI;
			double angle2 = angle + M_PI + 20.0 / 180.0 * M_PI;

			double x3 = x2 + 10 * cos(angle1);
			double y3 = y2 + 10 * sin(angle1);
			cv::line(result, cv::Point(x2, y2), cv::Point(x3, y3), cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

			double x4 = x2 + 10 * cos(angle2);
			double y4 = y2 + 10 * sin(angle2);
			cv::line(result, cv::Point(x2, y2), cv::Point(x4, y4), cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
		}
	}

	if (showVertices) {
		for (int i = 0; i < vertices.size(); ++i) {
			cv::Scalar color;
			if (vertices[i].second == 1) {
				color = cv::Scalar(255, 0, 0);
			}
			else if (vertices[i].second == 2) {
				color = cv::Scalar(0, 0, 255);
			}
			else {
				color = cv::Scalar(0, 255, 0);
			}
			cv::circle(result, cv::Point(vertices[i].first.x + size / 2, vertices[i].first.y + size / 2), 3, color, 1);
		}
	}

	cv::flip(result, result, 0);
	cv::imwrite(filename.c_str(), result);
}

void test(int size, double segment_length, double angle, double curvature, const std::string& filename) {
	std::vector<std::pair<glm::dvec2, int>> vertices;
	std::vector<std::pair<glm::dvec2, glm::dvec2>> edges;
	
	// １本の道路をとりあえず作成
	std::vector<std::pair<glm::dvec2, double>> regular_elements;
	vertices.push_back(std::make_pair(glm::dvec2(0, 0), 1));
	regular_elements.push_back(std::make_pair(glm::dvec2(0, 0), angle));
	growRoads(size, angle, glm::vec2(0, 0), curvature, segment_length, regular_elements, vertices, edges);
	growRoads(size, angle, glm::vec2(0, 0), curvature, -segment_length, regular_elements, vertices, edges);
	//saveRoadImage(size, vertices, edges, "initial.png", false, false);

	// setup the tensor field
	cv::Mat tensor(size, size, CV_64F);
	for (int r = 0; r < tensor.rows; ++r) {
		for (int c = 0; c < tensor.rows; ++c) {
			int x = c - tensor.rows / 2;
			int y = r - tensor.rows / 2;

			double total_angle = 0.0;
			double total_weight = 0.0;
			for (int k = 0; k < regular_elements.size(); ++k) {
				double dist = glm::length(glm::dvec2(x, y) - regular_elements[k].first);

				// convert the angle to be within [-pi/2, pi/2]
				double angle = regular_elements[k].second;
				double weight = exp(-dist / 10);
				total_angle += angle * weight;
				total_weight += weight;
			}

			float avg_angle = total_angle / total_weight;
			tensor.at<double>(r, c) = avg_angle;
		}
	}
	//saveTensorImage(tensor, "tensor.png");

	// generate roads
	generateRoadsByTensor(tensor, segment_length, segment_length * 0.7, vertices, edges);

	// visualize the roads
	saveRoadImage(size, vertices, edges, filename, false, false);
}

int main() {
	int size = 1000;

	srand(12);
	test(size, 80, 0.0, 0.1, "result1.png");
	srand(12345);
	test(size, 80, 0.3, 0.2, "result2.png");
	srand(1234578);
	test(size, 80, 0.6, 0.15, "result3.png");
	srand(125);
	test(size, 80, 0.2, 0.3, "result4.png");
	srand(5213);
	test(size, 80, 0.75, 0.1, "result5.png");
	srand(2352);
	test(size, 80, 0.45, 0.0, "result6.png");

	return 0;
}