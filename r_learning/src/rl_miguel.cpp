// binary_map_node.cpp
//
// ROS 1 (Noetic) node that:
//   • Receives an OccupancyGrid
//   • Converts it to a binary grid
//   • Trains a simple Q‑learning agent to go from the top‑left to bottom‑right
//   • Publishes a densified nav_msgs/Path
//
// Build (inside a catkin package):
//   add_executable(binary_map_node src/binary_map_node.cpp)
//   target_link_libraries(binary_map_node ${catkin_LIBRARIES})
//   add_dependencies(binary_map_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
//
// Author: ChatGPT (o3) – 2025‑07‑16
// ------------------------------------------------------------

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

#include <vector>
#include <deque>
#include <tuple>
#include <random>
#include <algorithm>
#include <cmath>
#include <string>

using std::pair;
using std::tuple;
using std::vector;

// ============================================================
//                         GridWorld
// ============================================================
class GridWorld {
public:
    int width{}, height{};
    vector<vector<int>> grid;                 // 0 = free, 1 = obstacle
    pair<int,int> start{0,0}, goal{0,0};
    vector<pair<int,int>> actions;            // 4‑conn or 8‑conn
    double reward_goal, reward_obstacle, reward_step, diag_cost;
    bool allow_diag;
    std::string shaping;

    GridWorld(const vector<vector<int>>& map,
              bool allow_diagonal   = false,
              const std::string& sh = "euclidean",
              double r_goal         = 100.0,
              double r_obs          = -10.0,
              double r_step         = -0.001)
        : grid(map),
          reward_goal(r_goal),
          reward_obstacle(r_obs),
          reward_step(r_step),
          allow_diag(allow_diagonal),
          shaping(sh)
    {
        height = static_cast<int>(grid.size());
        width  = static_cast<int>(grid.empty() ? 0 : grid[0].size());
        diag_cost = allow_diag ? std::sqrt(2.0) : 1.0;

        start = {0, 0};
        goal  = {height - 1, width - 1};

        // guarantee start/goal are free
        grid[start.first][start.second] = 0;
        grid[goal.first][goal.second]   = 0;

        if (allow_diag) {
            actions = {{-1,0},{1,0},{0,-1},{0,1},
                       {-1,-1},{-1,1},{1,-1},{1,1}};
        } else {
            actions = {{-1,0},{1,0},{0,-1},{0,1}};
        }
    }

    inline pair<int,int> reset() const { return start; }

    tuple<pair<int,int>, double, bool>
    step(const pair<int,int>& pos, int action) const
    {
        int r = pos.first, c = pos.second;
        int dr = actions[action].first;
        int dc = actions[action].second;
        int nr = r + dr, nc = c + dc;

        pair<int,int> next_state;
        double reward;
        bool done;

        if (nr < 0 || nr >= height || nc < 0 || nc >= width || grid[nr][nc] == 1)
        {
            next_state = {r, c};
            reward = reward_obstacle;
            done   = false;
        }
        else
        {
            next_state = {nr, nc};
            if (next_state == goal) {
                reward = reward_goal;
                done   = true;
            } else {
                double step_pen =
                    (allow_diag && dr && dc && reward_step < 0) ? -diag_cost
                                                                : reward_step;
                reward = step_pen;
                done   = false;
            }
        }

        // potential‑based shaping
        if (shaping == "euclidean" || shaping == "manhattan") {
            auto dist = [&](const pair<int,int>& s) {
                if (shaping == "manhattan")
                    return double(std::abs(s.first - goal.first) +
                                  std::abs(s.second - goal.second));
                return std::hypot(double(s.first - goal.first),
                                  double(s.second - goal.second));
            };
            reward += dist({r,c}) - dist(next_state);
        }

        return {next_state, reward, done};
    }
};

// ============================================================
//                       Q‑learning Agent
// ============================================================
class QLearningAgent {
public:
    GridWorld& env;
    double alpha, gamma, epsilon, min_eps, decay;
    std::string schedule;
    vector<vector<vector<double>>> Q;

    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<> uni{0.0,1.0};

    QLearningAgent(GridWorld& env_,
                   double a = 0.1,
                   double g = 0.99,
                   double e = 1.0,
                   double me = 0.05,
                   double d = 0.9,
                   const std::string& sched = "adaptive")
        : env(env_), alpha(a), gamma(g), epsilon(e), min_eps(me),
          decay(d), schedule(sched)
    {
        Q.resize(env.height,
                 vector<vector<double>>(env.width,
                   vector<double>(env.actions.size(), 0.0)));
    }

    int choose_action(const pair<int,int>& s) {
        if (uni(rng) < epsilon) {                       // explore
            std::uniform_int_distribution<> uid(0, int(env.actions.size())-1);
            return uid(rng);
        }
        // exploit
        auto [r,c] = s;
        const auto& q = Q[r][c];
        double best = *std::max_element(q.begin(), q.end());
        vector<int> idx;
        for (int i=0;i<int(q.size());++i)
            if (q[i] == best) idx.push_back(i);
        std::uniform_int_distribution<> uid(0, int(idx.size())-1);
        return idx[uid(rng)];
    }

    void update(const pair<int,int>& s, int a,
                double r, const pair<int,int>& s2, bool done)
    {
        auto [r0,c0] = s;
        auto [r1,c1] = s2;
        double target = done ? r : r + gamma * *std::max_element(Q[r1][c1].begin(),
                                                                 Q[r1][c1].end());
        Q[r0][c0][a] += alpha * (target - Q[r0][c0][a]);
    }

    void decay_epsilon() {
        epsilon = std::max(min_eps, epsilon * decay);
    }
};

// ============================================================
//                       Helper Functions
// ============================================================
void train(GridWorld& env, QLearningAgent& agent,
           int episodes = 500, int log_every = 50)
{
    for (int ep = 1; ep <= episodes; ++ep) {
        auto state = env.reset();
        bool done  = false;
        while (!done) {
            int a = agent.choose_action(state);
            auto [s2, r, is_done] = env.step(state, a);
            agent.update(state, a, r, s2, is_done);
            state = s2;
            done  = is_done;
        }
        agent.decay_epsilon();
        if (ep % log_every == 0) ROS_INFO_STREAM("Episode " << ep << "/" << episodes);
    }
}

nav_msgs::Path
densifyPath(const vector<pair<int,int>>& cells,
            double resolution,
            double pts_per_m,
            const std::string& frame_id = "map")
{
    nav_msgs::Path path;
    path.header.frame_id = frame_id;

    if (cells.size() < 2) return path;

    for (size_t i = 0; i + 1 < cells.size(); ++i) {
        auto [r0,c0] = cells[i];
        auto [r1,c1] = cells[i+1];

        double x0 = (c0 + 0.5) * resolution;
        double y0 = (r0 + 0.5) * resolution;
        double x1 = (c1 + 0.5) * resolution;
        double y1 = (r1 + 0.5) * resolution;

        double dx = x1 - x0;
        double dy = y1 - y0;
        double seg_len = std::hypot(dx, dy);
        double yaw = std::atan2(dy, dx);

        int n_pts = std::max(1, int(seg_len * pts_per_m));
        for (int k = 0; k < n_pts; ++k) {
            double t = double(k) / n_pts;
            geometry_msgs::PoseStamped p;
            p.pose.position.x = x0 + t*dx;
            p.pose.position.y = y0 + t*dy;
            p.pose.orientation.z = std::sin(yaw*0.5);
            p.pose.orientation.w = std::cos(yaw*0.5);
            path.poses.push_back(p);
        }
    }
    // append final cell center
    auto [rg,cg] = cells.back();
    geometry_msgs::PoseStamped p;
    p.pose.position.x = (cg + 0.5) * resolution;
    p.pose.position.y = (rg + 0.5) * resolution;
    p.pose.orientation.w = 1.0;
    path.poses.push_back(p);

    return path;
}

// ============================================================
//                   BinaryMapConverter Node
// ============================================================
class BinaryMapConverter {
    ros::NodeHandle nh_, pnh_;
    ros::Subscriber map_sub_;
    ros::Publisher  path_pub_;

    double occ_thresh_, pts_per_m_;
    bool unknown_is_free_;
    double map_resolution_{0.05};

    std::mt19937 rng_{std::random_device{}()};

public:
    BinaryMapConverter()
        : pnh_("~")
    {
        pnh_.param("occ_threshold",    occ_thresh_,      50.0);
        pnh_.param("points_per_meter", pts_per_m_,       20.0);
        pnh_.param("unknown_is_free",  unknown_is_free_, true);

        std::string map_topic = pnh_.param<std::string>("map_topic",  "/map");
        std::string path_topic= pnh_.param<std::string>("path_topic", "/path");

        path_pub_ = nh_.advertise<nav_msgs::Path>(path_topic, 10);
        map_sub_  = nh_.subscribe(map_topic, 1,
                       &BinaryMapConverter::mapCallback, this);

        ROS_INFO_STREAM("[binary_map_node] Subscribed to " << map_topic
                        << ", publishing path on " << path_topic);
    }

private:
    void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg)
    {
        int W = msg->info.width;
        int H = msg->info.height;
        map_resolution_ = msg->info.resolution;

        // Convert to binary grid
        vector<vector<int>> grid(H, vector<int>(W, 0));
        for (int r=0; r<H; ++r) {
            for (int c=0; c<W; ++c) {
                int v = msg->data[r*W + c];
                if (v == -1) v = unknown_is_free_ ? 0 : 100;
                grid[r][c] = (v >= occ_thresh_) ? 1 : 0;
            }
        }
        ROS_INFO_STREAM("Map received (" << W << "x" << H << "). Training…");

        // Build environment & agent
        GridWorld env(grid, /*allow_diagonal=*/false, "euclidean", 100.0, -10.0, -0.0001);
        QLearningAgent agent(env, 0.1, 0.99, 1.0, 0.05, 0.9, "mix");
        train(env, agent, /*episodes=*/100, /*log_every=*/20);

        // Greedy path extraction
        vector<pair<int,int>> cell_path;
        agent.epsilon = 0.0;                     // purely greedy
        auto s = env.reset();
        cell_path.push_back(s);
        for (int step=0; step<1000; ++step) {
            if (s == env.goal) break;
            int a = agent.choose_action(s);
            auto [s2, _, done] = env.step(s, a);
            cell_path.push_back(s2);
            s = s2;
            if (done) break;
        }

        // Densify & publish
        auto path_msg = densifyPath(cell_path,
                                    map_resolution_,
                                    pts_per_m_,
                                    msg->header.frame_id);
        path_msg.header.stamp = ros::Time::now();
        path_pub_.publish(path_msg);
        ROS_INFO_STREAM("Path with " << path_msg.poses.size() << " poses published.");
    }
};

// ============================================================
//                             main
// ============================================================
int main(int argc, char** argv)
{
    ros::init(argc, argv, "binary_map_node_cpp");
    BinaryMapConverter node;
    ros::spin();
    return 0;
}
