import * as fs from "fs";
import * as path from "path";

export async function readAdditionalContextFiles(filePaths: string[]): Promise<string[]> {
    const contextContents: string[] = [];
    if (!filePaths) {
        return [];
    }

    // Here we assume the file paths are all ok,
    // let the caller handle the error if there is any to avoid silence error
    for (const file of filePaths) {
        const filePath = path.isAbsolute(file) ? file : path.resolve(process.cwd(), file.trim());
        const content = fs.readFileSync(filePath, "utf-8");
        contextContents.push(content);
    }

    return contextContents;
}
