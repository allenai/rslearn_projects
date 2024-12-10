package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/mitroadmaps/gomapinfer/common"
)

const FUTURE_LABEL = "2030-01"
const TILE_SIZE = 2048

// Don't consider groups with fewer than this many valid timesteps.
// Note that the point doesn't need to be detected in all the timesteps, this is just
// timesteps where we have image coverage.
const MIN_VALID_TIMESTEPS = 8

type Tile struct {
	Projection string
	Column     int
	Row        int
}

type Point struct {
	Type     string `json:"type"`
	Geometry struct {
		Type        string     `json:"type"`
		Coordinates [2]float64 `json:"coordinates"`
	} `json:"geometry"`
	label      string
	Properties struct {
		Category   *string  `json:"category"`
		Score      *float64 `json:"score"`
		Projection *string  `json:"projection,omitempty"`
		Column     *int     `json:"col,omitempty"`
		Row        *int     `json:"row,omitempty"`
		Start      string   `json:"start,omitempty"`
		End        string   `json:"end,omitempty"`
	} `json:"properties"`
}

type PointData struct {
	Type       string  `json:"type"`
	Features   []Point `json:"features"`
	Properties struct {
		ValidPatches map[string][][2]int `json:"valid_patches,omitempty"`
	} `json:"properties"`
}

type Group []Point

func (g Group) Center() [2]int {
	var sum [2]int
	for _, p := range g {
		sum[0] += *p.Properties.Column
		sum[1] += *p.Properties.Row
	}
	return [2]int{
		sum[0] / len(g),
		sum[1] / len(g),
	}
}

func decrementLabel(label string) string {
	parts := strings.Split(label, "-")
	year, _ := strconv.Atoi(parts[0])
	month, _ := strconv.Atoi(parts[1])
	month -= 1
	if month == 0 {
		year -= 1
		month = 12
	}
	return fmt.Sprintf("%04d-%02d", year, month)
}

// Returns the end label for idx.
// If idx >= len(labels), we return a label far in the future.
// Otherwise, we just return labels[idx].
func getEndLabel(labels []string, idx int) string {
	if idx >= len(labels) {
		return FUTURE_LABEL
	}
	return labels[idx]
}

// Grid size which must be larger than the maximum expected distance threshold.
// This is currently in zoom 13 pixel coordinates so each unit is about 10 m.
const GridSize float64 = 256

// Factor to divide by to convert meters to point units.
const MetersPerPixel = 10

func main() {
	labels := flag.String("labels", "", "Comma-separated list of labels")
	pointFname := flag.String("fname", "", "Point filename with LABEL placeholder like in/LABEL.geojson")
	outFname := flag.String("out", "", "Output filename with LABEL placeholder like out/LABEL.geojson")
	histFname := flag.String("hist", "", "Merged history output filename")
	distanceThreshold := flag.Float64("max_dist", 200, "Matching distance threshold in meters")
	nmsDistance := flag.Float64("nms_dist", 200.0/111111, "NMS distance in degrees")
	numThreads := flag.Int("threads", 32, "Number of threads")
	flag.Parse()

	labelList := strings.Split(*labels, ",")

	// Read points beginning with the most recent set
	// (which is likely the one that covers the most points).
	var groups []Group
	// Keep track of map from tiles to labels in which the tile is valid.
	tileLabelValidity := make(map[Tile][]string)
	for labelIdx := len(labelList) - 1; labelIdx >= 0; labelIdx-- {
		label := labelList[labelIdx]

		fname := strings.ReplaceAll(*pointFname, "LABEL", label)
		if _, err := os.Stat(fname); os.IsNotExist(err) {
			continue
		}
		bytes, err := os.ReadFile(fname)
		if err != nil {
			panic(err)
		}

		var data PointData
		if err := json.Unmarshal(bytes, &data); err != nil {
			panic(err)
		}

		// Build grid index from the current features.
		curPoints := data.Features
		for idx := range curPoints {
			curPoints[idx].label = label
		}
		gridIndexes := make(map[string]*common.GridIndex)
		for idx, point := range curPoints {
			projection := *point.Properties.Projection
			col := float64(*point.Properties.Column)
			row := float64(*point.Properties.Row)
			if gridIndexes[projection] == nil {
				gridIndexes[projection] = common.NewGridIndex(GridSize)
			}
			gridIndexes[projection].Insert(idx, common.Rectangle{
				Min: common.Point{col, row},
				Max: common.Point{col, row},
			})
		}

		log.Printf("matching %d groups with %d features at %v", len(groups), len(curPoints), label)

		// Match existing groups to the new points.
		matchedIndices := make(map[int]bool)
		for groupIdx, group := range groups {
			projection := *group[0].Properties.Projection
			center := group.Center()

			// Lookup candidate new points that could match this group using the grid index.
			var indices []int
			if gridIndexes[projection] != nil {
				indices = gridIndexes[projection].Search(common.Rectangle{
					Min: common.Point{float64(center[0]) - GridSize, float64(center[1]) - GridSize},
					Max: common.Point{float64(center[0]) + GridSize, float64(center[1]) + GridSize},
				})
			}

			var closestIdx int = -1
			var closestDistance float64
			for _, idx := range indices {
				if matchedIndices[idx] {
					continue
				}

				// Double check distance threshold since the index may still return
				// points that are slightly outside the threshold.
				// We used to check category too, but now we use the category of the
				// last prediction, and just apply a distance penalty for mismatched
				// category, since we noticed that sometimes there are partially
				// constructed wind turbines detected as platforms but then later
				// detected as turbines once construction is done, and we don't want
				// that to mess up the Viterbi smoothing. Put another way, marine
				// infrastructure should show up in our map even if we're not exactly
				// sure about the category.
				dx := center[0] - *curPoints[idx].Properties.Column
				dy := center[1] - *curPoints[idx].Properties.Row
				distance := math.Sqrt(float64(dx*dx + dy*dy))

				if distance > *distanceThreshold/MetersPerPixel {
					continue
				}

				if *group[0].Properties.Category != *curPoints[idx].Properties.Category {
					distance += *distanceThreshold / MetersPerPixel
				}

				if closestIdx == -1 || distance < closestDistance {
					closestIdx = idx
					closestDistance = distance
				}
			}

			if closestIdx == -1 {
				continue
			}

			matchedIndices[closestIdx] = true
			groups[groupIdx] = append(groups[groupIdx], curPoints[closestIdx])
		}

		// Add unmatched points in the current time as new groups.
		for idx, point := range curPoints {
			if matchedIndices[idx] {
				continue
			}
			groups = append(groups, Group{point})
		}

		// Also update valid tiles/labels.
		for projection, patches := range data.Properties.ValidPatches {
			for _, patch := range patches {
				tile := Tile{
					Projection: projection,
					Column:     patch[0],
					Row:        patch[1],
				}
				tileLabelValidity[tile] = append(tileLabelValidity[tile], label)
			}
		}
	}

	// Apply non-maximal suppression over groups.
	// We prefer longer groups, or if they are the same length, the group with higher
	// last score.
	log.Println("applying non-maximal suppression")
	nmsIndex := common.NewGridIndex(*nmsDistance * 5)
	for groupIdx, group := range groups {
		last := group[len(group)-1]
		coordinates := last.Geometry.Coordinates
		nmsIndex.Insert(groupIdx, common.Point{coordinates[0], coordinates[1]}.Rectangle())
	}
	var newGroups []Group
	for groupIdx, group := range groups {
		last := group[len(group)-1]
		coordinates := last.Geometry.Coordinates
		results := nmsIndex.Search(common.Point{coordinates[0], coordinates[1]}.RectangleTol(*nmsDistance))
		needsRemoval := false
		for _, otherIdx := range results {
			if otherIdx == groupIdx {
				continue
			}
			other := groups[otherIdx]
			otherLast := other[len(other)-1]
			otherCoordinates := otherLast.Geometry.Coordinates
			dx := coordinates[0] - otherCoordinates[0]
			dy := coordinates[1] - otherCoordinates[1]
			distance := math.Sqrt(float64(dx*dx + dy*dy))
			if distance >= *nmsDistance {
				continue
			}

			// It is within distance threshold, so see if group is worse than other.
			if len(group) < len(other) {
				needsRemoval = true
			} else if len(group) == len(other) && *last.Properties.Score < *otherLast.Properties.Score {
				needsRemoval = true
			}
		}

		if !needsRemoval {
			newGroups = append(newGroups, group)
		}
	}
	log.Printf("NMS filtered from %d to %d groups", len(groups), len(newGroups))
	groups = newGroups

	// Apply Viterbi algorithm in each group.
	initialProbs := []float64{0.5, 0.5}
	transitionProbs := [][]float64{
		{0.95, 0.05},
		{0.01, 0.99},
	}
	emissionProbs := [][]float64{
		{0.8, 0.2},
		{0.2, 0.8},
	}
	// Convert as observation history to a list of ranges when the state is non-zero.
	applyViterbi := func(history []int) [][2]int {
		probs := make([]float64, len(initialProbs))
		copy(probs, initialProbs)
		var pointers [][]int

		// Forward pass.
		for _, observation := range history {
			newProbs := make([]float64, len(probs))
			curPointers := make([]int, len(probs))
			// For each new state, take max over probability resulting from different prev states.
			for newState := range probs {
				for prevState, prevProb := range probs {
					prob := prevProb * transitionProbs[prevState][newState] * emissionProbs[newState][observation]
					if prob > newProbs[newState] {
						newProbs[newState] = prob
						curPointers[newState] = prevState
					}
				}
			}
			probs = newProbs
			pointers = append(pointers, curPointers)
		}

		// Backward pass: compute max and then follow the pointers.
		var finalState int
		var bestProb float64
		for state, prob := range probs {
			if prob < bestProb {
				continue
			}
			bestProb = prob
			finalState = state
		}
		reversedStates := []int{finalState}
		curState := finalState
		for i := len(pointers) - 1; i > 0; i-- {
			curState = pointers[i][curState]
			reversedStates = append(reversedStates, curState)
		}
		states := make([]int, len(reversedStates))
		for i := range states {
			states[i] = reversedStates[len(states)-i-1]
		}

		// Convert to ranges.
		var ranges [][2]int
		var startIdx int = -1
		for idx, state := range states {
			if state == 0 && startIdx >= 0 {
				// Object was active but no longer.
				ranges = append(ranges, [2]int{startIdx, idx})
				startIdx = -1
			}
			if state == 1 && startIdx == -1 {
				startIdx = idx
			}
		}
		// Add last range if any.
		if startIdx != -1 {
			ranges = append(ranges, [2]int{startIdx, len(states)})
		}
		return ranges
	}

	// Pass each group through Viterbi algorithm.
	// This yields time ranges where a group was present in the world.
	// Usually there should just be one time range associated with each group,
	// but there could be multiple if there really was a gap.
	// Anyway we then collect those ranges into output data.
	var historyData PointData
	outFeatures := make(map[string]*PointData)
	log.Println("processing groups")
	ch := make(chan Group)
	type Rng struct {
		Group    Group
		StartIdx int
		EndIdx   int
	}
	donech := make(chan []Rng)
	for i := 0; i < *numThreads; i++ {
		go func() {
			var myRngs []Rng
			for group := range ch {
				// Create set of labels where the point is present.
				labelSet := make(map[string]bool)
				for _, point := range group {
					labelSet[point.label] = true
				}

				// Also create label set where the tile containing the point was valid.
				// If tile is invalid at a label, it implies there was no satellite image data at that location/time.
				validLabelSet := make(map[string]bool)
				center := group.Center()
				tile := Tile{
					Projection: *group[0].Properties.Projection,
					Column:     int(math.Floor(float64(center[0]) / TILE_SIZE)),
					Row:        int(math.Floor(float64(center[1]) / TILE_SIZE)),
				}
				for _, label := range tileLabelValidity[tile] {
					validLabelSet[label] = true
				}

				if len(validLabelSet) < MIN_VALID_TIMESTEPS {
					continue
				}

				// Now make history of observations for Viterbi algorithm.
				// We only include timesteps where the tile was valid.
				// We also create a map from observed timesteps to original timestep index.
				var observations []int
				var labelIdxMap []int
				for labelIdx, label := range labelList {
					if !validLabelSet[label] {
						continue
					}
					labelIdxMap = append(labelIdxMap, labelIdx)
					if labelSet[label] {
						observations = append(observations, 1)
					} else {
						observations = append(observations, 0)
					}
				}

				ranges := applyViterbi(observations)
				for _, rng := range ranges {
					startIdx := labelIdxMap[rng[0]]
					var endIdx int
					if rng[1] == len(observations) {
						endIdx = len(labelList)
					} else {
						endIdx = labelIdxMap[rng[1]]
					}
					myRngs = append(myRngs, Rng{
						Group:    group,
						StartIdx: startIdx,
						EndIdx:   endIdx,
					})
				}
			}
			donech <- myRngs
		}()
	}
	for _, group := range groups {
		ch <- group
	}
	close(ch)
	for i := 0; i < *numThreads; i++ {
		curRngs := <-donech
		for _, rng := range curRngs {
			last := rng.Group[len(rng.Group)-1]
			feat := Point{}
			feat.Type = "Feature"
			feat.Geometry = last.Geometry
			feat.Properties.Category = last.Properties.Category
			feat.Properties.Score = last.Properties.Score

			// Add the feature to the monthly outputs.
			for labelIdx := rng.StartIdx; labelIdx < rng.EndIdx; labelIdx++ {
				label := labelList[labelIdx]
				if outFeatures[label] == nil {
					outFeatures[label] = &PointData{
						Type: "FeatureCollection",
					}
				}
				outFeatures[label].Features = append(outFeatures[label].Features, feat)
			}

			// Now set start and end label for this feature correctly.
			// Along with the score (computed as average of the points in the group).
			// And then add it to history.
			feat.Properties.Start = labelList[rng.StartIdx]
			feat.Properties.End = getEndLabel(labelList, rng.EndIdx)

			var scoreSum float64 = 0
			for _, p := range rng.Group {
				scoreSum += *p.Properties.Score
			}
			scoreAvg := scoreSum / float64(len(rng.Group))
			feat.Properties.Score = &scoreAvg

			historyData.Features = append(historyData.Features, feat)
		}
	}

	log.Println("writing outputs")

	if *histFname != "" {
		bytes, err := json.Marshal(historyData)
		if err != nil {
			panic(err)
		}
		if err := os.WriteFile(*histFname, bytes, 0644); err != nil {
			panic(err)
		}
	}

	if *outFname != "" {
		for label, data := range outFeatures {
			fname := strings.ReplaceAll(*outFname, "LABEL", label)
			bytes, err := json.Marshal(data)
			if err != nil {
				panic(err)
			}
			if err := os.WriteFile(fname, bytes, 0644); err != nil {
				panic(err)
			}
		}
	}
}
