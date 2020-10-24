#include "CanvasSystem.h"

#include "../Components/CanvasData.h"

#include <_deps/imgui/imgui.h>
#include <Eigen/Core>
#include <Eigen/LU>

using namespace Ubpa;

void CanvasSystem::OnUpdate(Ubpa::UECS::Schedule& schedule) {
	schedule.RegisterCommand([](Ubpa::UECS::World* w) {
		auto data = w->entityMngr.GetSingleton<CanvasData>();
		if (!data)
			return;

		if (ImGui::Begin("Canvas")) {
			ImGui::Checkbox("Enable grid", &data->opt_enable_grid);
			ImGui::Checkbox("Enable context menu", &data->opt_enable_context_menu);
			ImGui::Text("Mouse Left: click to add points,\nMouse Right: drag to scroll, click for context menu.");

			ImGui::SliderInt("Fitting Step", &data->opt_fitting_step, 1, 50);

			ImGui::CheckboxFlags("Lagrange Polynomial (Red)", &data->opt_fitting_methods, 1);
			ImGui::CheckboxFlags("Gaussian Interpolation (Green)", &data->opt_fitting_methods, 2);
			if (data->opt_fitting_methods & 2) {
				ImGui::SliderFloat("Gaussian Sigma", &data->opt_sigma, 0.001f, 1.0f);
			}
			ImGui::CheckboxFlags("Least Squares (Blue)", &data->opt_fitting_methods, 4);
			ImGui::CheckboxFlags("Ridge Regression (Yellow)", &data->opt_fitting_methods, 8);
			if (data->opt_fitting_methods & 12) {
				ImGui::SliderInt("Max Exponent", &data->opt_max_exponent, 1, 50);
			}
			if (data->opt_fitting_methods & 8) {
				ImGui::SliderFloat("Ridge Regression Lambda", &data->opt_lambda, 0.f, 1.f);
			}

			// Typically you would use a BeginChild()/EndChild() pair to benefit from a clipping region + own scrolling.
			// Here we demonstrate that this can be replaced by simple offsetting + custom drawing + PushClipRect/PopClipRect() calls.
			// To use a child window instead we could use, e.g:
			//      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));      // Disable padding
			//      ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(50, 50, 50, 255));  // Set a background color
			//      ImGui::BeginChild("canvas", ImVec2(0.0f, 0.0f), true, ImGuiWindowFlags_NoMove);
			//      ImGui::PopStyleColor();
			//      ImGui::PopStyleVar();
			//      [...]
			//      ImGui::EndChild();

			// Using InvisibleButton() as a convenience 1) it will advance the layout cursor and 2) allows us to use IsItemHovered()/IsItemActive()
			ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();      // ImDrawList API uses screen coordinates!
			ImVec2 canvas_sz = ImGui::GetContentRegionAvail();   // Resize canvas to what's available
			if (canvas_sz.x < 50.0f) canvas_sz.x = 50.0f;
			if (canvas_sz.y < 50.0f) canvas_sz.y = 50.0f;
			ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);

			// Draw border and background color
			ImGuiIO& io = ImGui::GetIO();
			ImDrawList* draw_list = ImGui::GetWindowDrawList();
			draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(50, 50, 50, 255));
			draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

			// This will catch our interactions
			ImGui::InvisibleButton("canvas", canvas_sz, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
			const bool is_hovered = ImGui::IsItemHovered(); // Hovered
			const bool is_active = ImGui::IsItemActive();   // Held
			const ImVec2 origin(canvas_p0.x + data->scrolling[0], canvas_p0.y + data->scrolling[1]); // Lock scrolled origin

			// use normalized position to reduce numerical errors
			const pointf2 mouse_pos_in_canvas((io.MousePos.x - origin.x) / canvas_sz.x, (io.MousePos.y - origin.y) / canvas_sz.y);

			// Add input point
			if (is_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
				data->input_points.push_back(mouse_pos_in_canvas);
			}

			// Clear output lists
			data->point_set1.clear();
			data->point_set2.clear();
			data->point_set3.clear();
			data->point_set4.clear();

			if (data->input_points.size()) {

				// Lagrange Polynomial
				if (data->opt_fitting_methods & 1) {
					for (float targetX = 0; targetX < 1.f; targetX += data->opt_fitting_step / canvas_sz.x) {
						pointf2 fittedPoint(targetX, 0.f);
						for (size_t j = 0; j < data->input_points.size(); j++) {
							float lj = 1.f;
							for (size_t m = 0; m < data->input_points.size(); m++) {
								if (m == j) continue;
								lj *= (targetX - data->input_points[m][0]) / (data->input_points[j][0] - data->input_points[m][0]);
							}
							fittedPoint[1] += data->input_points[j][1] * lj;
						}
						data->point_set1.push_back(fittedPoint);
					}
				}

				// Gaussian Interpolation
				if (data->opt_fitting_methods & 2) {
					// custom constrain: use the average y as b0
					float avgVal = 0.f;
					for (size_t j = 0; j < data->input_points.size(); j++) {
						avgVal += data->input_points[j][1];
					}
					avgVal /= data->input_points.size();

					float sigma = data->opt_sigma;
					Eigen::VectorXf Y(data->input_points.size());
					Eigen::MatrixXf G(data->input_points.size(), data->input_points.size());
					for (size_t i = 0; i < data->input_points.size(); i++) {
						Y(i) = data->input_points[i][1] - avgVal;
						for (size_t j = 0; j < data->input_points.size(); j++) {
							float dx = data->input_points[i][0] - data->input_points[j][0];
							G(i, j) = exp(-dx * dx / (2.f * sigma * sigma));
						}
					}
					Eigen::VectorXf B = G.inverse() * Y;

					for (float targetX = 0; targetX < 1.f; targetX += data->opt_fitting_step / canvas_sz.x) {
						pointf2 fittedPoint(targetX, avgVal);
						for (size_t i = 0; i < data->input_points.size(); i++) {
							float dx = targetX - data->input_points[i][0];
							fittedPoint[1] += B(i) * exp(-dx * dx / (2.f * sigma * sigma));
						}
						data->point_set2.push_back(fittedPoint);
					}
				}

				// Least Squares
				if (data->opt_fitting_methods & 4) {
					Eigen::VectorXf Y(data->input_points.size());
					Eigen::MatrixXf M(data->input_points.size(), data->opt_max_exponent);
					for (size_t i = 0; i < data->input_points.size(); i++) {
						Y(i) = data->input_points[i][1];
						float x = data->input_points[i][0]; M(i, 0) = 1.f;
						for (size_t j = 1; j < data->opt_max_exponent; j++) {
							M(i, j) = M(i, j - 1) * x;
						}
					}
					Eigen::MatrixXf Mt = M.transpose();
					Eigen::VectorXf A = (Mt * M).inverse() * (Mt * Y);

					for (float targetX = 0; targetX < 1.f; targetX += data->opt_fitting_step / canvas_sz.x) {
						pointf2 fittedPoint(targetX, 0.f);
						float powerTerm = 1.f;
						for (size_t i = 0; i < data->opt_max_exponent; i++) {
							fittedPoint[1] += A(i) * powerTerm;
							powerTerm *= targetX;
						}
						data->point_set3.push_back(fittedPoint);
					}
				}

				// Ridge Regression
				if (data->opt_fitting_methods & 8) {
					Eigen::VectorXf Y(data->input_points.size());
					Eigen::MatrixXf M(data->input_points.size(), data->opt_max_exponent);
					for (size_t i = 0; i < data->input_points.size(); i++) {
						Y(i) = data->input_points[i][1];
						float x = data->input_points[i][0]; M(i, 0) = 1.f;
						for (size_t j = 1; j < data->opt_max_exponent; j++) {
							M(i, j) = M(i, j - 1) * x;
						}
					}
					Eigen::MatrixXf Mt = M.transpose();
					Eigen::MatrixXf R = Eigen::MatrixXf::Identity(data->opt_max_exponent, data->opt_max_exponent) * data->opt_lambda;
					Eigen::VectorXf A = (Mt * M + R).inverse() * (Mt * Y);

					for (float targetX = 0; targetX < 1.f; targetX += data->opt_fitting_step / canvas_sz.x) {
						pointf2 fittedPoint(targetX, 0.f);
						float powerTerm = 1.f;
						for (size_t i = 0; i < data->opt_max_exponent; i++) {
							fittedPoint[1] += A(i) * powerTerm;
							powerTerm *= targetX;
						}
						data->point_set4.push_back(fittedPoint);
					}
				}
			}

			// Pan (we use a zero mouse threshold when there's no context menu)
			// You may decide to make that threshold dynamic based on whether the mouse is hovering something etc.
			const float mouse_threshold_for_pan = data->opt_enable_context_menu ? -1.0f : 0.0f;
			if (is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Right, mouse_threshold_for_pan)) {
				data->scrolling[0] += io.MouseDelta.x;
				data->scrolling[1] += io.MouseDelta.y;
			}

			// Context menu (under default mouse threshold)
			ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
			if (data->opt_enable_context_menu && ImGui::IsMouseReleased(ImGuiMouseButton_Right) && drag_delta.x == 0.0f && drag_delta.y == 0.0f)
				ImGui::OpenPopupContextItem("context");
			if (ImGui::BeginPopup("context")) {
				if (ImGui::MenuItem("Remove one", NULL, false, data->input_points.size() > 0)) { data->input_points.resize(data->input_points.size() - 1); }
				if (ImGui::MenuItem("Remove all", NULL, false, data->input_points.size() > 0)) { data->input_points.clear(); }
				ImGui::EndPopup();
			}

			// Draw grid + all lines in the canvas
			draw_list->PushClipRect(canvas_p0, canvas_p1, true);
			if (data->opt_enable_grid) {
				const float GRID_STEP = 64.0f;
				for (float x = fmodf(data->scrolling[0], GRID_STEP); x < canvas_sz.x; x += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x + x, canvas_p0.y), ImVec2(canvas_p0.x + x, canvas_p1.y), IM_COL32(200, 200, 200, 40));
				for (float y = fmodf(data->scrolling[1], GRID_STEP); y < canvas_sz.y; y += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x, canvas_p0.y + y), ImVec2(canvas_p1.x, canvas_p0.y + y), IM_COL32(200, 200, 200, 40));
			}

			for (size_t n = 0; data->point_set1.size() && n < data->point_set1.size() - 1; n++) {
				pointf2& p1 = data->point_set1[n], p2 = data->point_set1[n + 1];
				draw_list->AddLine(
					ImVec2(origin.x + p1[0] * canvas_sz.x, origin.y + p1[1] * canvas_sz.y),
					ImVec2(origin.x + p2[0] * canvas_sz.x, origin.y + p2[1] * canvas_sz.y),
					IM_COL32(255, 0, 0, 255), 2.0f);
			}
			for (size_t n = 0; data->point_set2.size() && n < data->point_set2.size() - 1; n++) {
				pointf2& p1 = data->point_set2[n], p2 = data->point_set2[n + 1];
				draw_list->AddLine(
					ImVec2(origin.x + p1[0] * canvas_sz.x, origin.y + p1[1] * canvas_sz.y),
					ImVec2(origin.x + p2[0] * canvas_sz.x, origin.y + p2[1] * canvas_sz.y),
					IM_COL32(0, 255, 0, 255), 2.0f);
			}
			for (size_t n = 0; data->point_set3.size() && n < data->point_set3.size() - 1; n++) {
				pointf2& p1 = data->point_set3[n], p2 = data->point_set3[n + 1];
				draw_list->AddLine(
					ImVec2(origin.x + p1[0] * canvas_sz.x, origin.y + p1[1] * canvas_sz.y),
					ImVec2(origin.x + p2[0] * canvas_sz.x, origin.y + p2[1] * canvas_sz.y),
					IM_COL32(0, 0, 255, 255), 2.0f);
			}
			for (size_t n = 0; data->point_set4.size() && n < data->point_set4.size() - 1; n++) {
				pointf2& p1 = data->point_set4[n], p2 = data->point_set4[n + 1];
				draw_list->AddLine(
					ImVec2(origin.x + p1[0] * canvas_sz.x, origin.y + p1[1] * canvas_sz.y),
					ImVec2(origin.x + p2[0] * canvas_sz.x, origin.y + p2[1] * canvas_sz.y),
					IM_COL32(255, 255, 0, 255), 2.0f);
			}

			for (size_t n = 0; n < data->input_points.size(); n++) {
				pointf2& p = data->input_points[n];
				draw_list->AddCircleFilled(
					ImVec2(origin.x + p[0] * canvas_sz.x, origin.y + p[1] * canvas_sz.y),
					3.0f, IM_COL32(255, 255, 255, 255));
			}

			draw_list->PopClipRect();
		}

		ImGui::End();
	});
}
