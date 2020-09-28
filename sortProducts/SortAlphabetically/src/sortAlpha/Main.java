package sortAlpha;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

public class Main {
	static List<String> lines;

	public static void main(String[] args) throws IOException {
			lines = Files.readAllLines(Paths.get("C:/Users/Prog/Desktop/GroceryListPredictor/webapp/allproducts2.txt"));
		Set<String> list = new TreeSet<String>(lines);
		FileWriter writer = new FileWriter("C:/Users/Prog/Desktop/GroceryListPredictor/webapp/sortedProducts.txt");
		for (String str : list) {
			writer.write(str + System.lineSeparator());
		}
		writer.close();
	}
}
